import json
import re
import logging
from typing import Any, List, TypedDict, Annotated

from enum import Enum
from arkitekt_next import register, easy, aprogress
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from rekuest_next.definition.utils import DescriptionAddin
from alpaka.api.schema import (
    LLMModel,
    Message,
    Room,
    Role,
    StructureInput,
    achat,
    ChatMessageInput,
    FunctionCallInput,
    FunctionDefinitionInput,
    ToolCallInput,
    ToolInput,
    ToolType,
    asend,
)
from mikro_next.api.schema import Image, aget_image
from rekuest_next import acall_raw
from rekuest_next.api.schema import (
    ActionFilter,
    OffsetPaginationInput as RekuestOffsetPaginationInput,
    PortKind,
    afind,
    alist_actions,
    aprimary_actions,
)
from rekuest_next.contrib.fastapi.openapi_utils import create_json_schema_from_ports
from rekuest_next.structures.default import get_default_structure_registry
from rekuest_next.structures.serialization.postman import aexpand_returns

# Configure logging
logging.basicConfig(level=logging.INFO, format="[chat] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_ACTION_TOOL_NAME = "search_arkitekt_actions"
ACTION_TOOL_PREFIX = "arkitekt_action"
MAX_TOOL_ROUNDS = 6

SEARCH_ACTION_TOOL = ToolInput(
    type=ToolType.FUNCTION,
    function=FunctionDefinitionInput(
        name=SEARCH_ACTION_TOOL_NAME,
        description="Search the Arkitekt server for actions to help with the user's task.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "identifier": {
                    "type": "string",
                    "description": "Optional structure identifier, e.g., @mikro/image",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max actions to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
)
TOOLS = (SEARCH_ACTION_TOOL,)

SYSTEM_PROMPT = (
    "You are a helpful assistant for Arkitekt and microscopy workflows. "
    "Give direct, useful answers, ask a clarifying question when ambiguous, and use attached images. "
    "Be specific: mention action names, args used, and return types. "
    "Use search_arkitekt_actions to find solutions. For image tasks, prefer @mikro/image. "
    "Do not claim to see pixel content that is not explicitly in metadata."
)


def preview_text(value: str | None, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit] + (
        "..." if value and len(value) > limit else ""
    )


def normalize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_value(i) for i in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


# -------------------------------------------------------------------------
# 1. State Definition & Reducer
# -------------------------------------------------------------------------
def manage_history(
    existing: List[ChatMessageInput], new: List[ChatMessageInput]
) -> List[ChatMessageInput]:
    combined = (existing or []) + new
    return [m for m in combined if m.role == Role.SYSTEM] + [
        m for m in combined if m.role != Role.SYSTEM
    ][-15:]


class AgentState(TypedDict):
    room: Room
    model: LLMModel
    image: Image | None
    available_action_ids: List[str]
    pending_structures: List[StructureInput]
    messages: Annotated[List[ChatMessageInput], manage_history]
    latest_response: str
    sent_message: Message | None


def is_autofillable_port(port, image: Image | None) -> bool:
    return image is not None and getattr(port, "identifier", None) == "@mikro/image"


def action_tool_name(action) -> str:
    slug = (
        re.sub(r"[^a-zA-Z0-9]+", "_", action.name).strip("_").lower()[:30] or "action"
    )
    return f"{ACTION_TOOL_PREFIX}_{slug}_{str(action.id).replace('-', '_')[:8]}"


def action_to_tool(action, image: Image | None) -> ToolInput:
    schema = create_json_schema_from_ports(
        action.args, f"{action_tool_name(action)}_args"
    )
    required = list(schema.get("required", []))

    for port in action.args:
        if is_autofillable_port(port, image):
            prop = schema["properties"].setdefault(port.key, {})
            prop["description"] = (
                f"{prop.get('description', '')} Auto-filled from attached image.".strip()
            )
            if port.key in required:
                required.remove(port.key)

    if required:
        schema["required"] = required
    else:
        schema.pop("required", None)

    return ToolInput(
        type=ToolType.FUNCTION,
        function=FunctionDefinitionInput(
            name=action_tool_name(action),
            description=action.description or f"Execute {action.name}.",
            parameters=schema,
        ),
    )


async def get_actions_by_ids(action_ids: List[str]) -> List[Any]:
    if not action_ids:
        return []
    actions = await alist_actions(
        filters=ActionFilter(ids=tuple(action_ids)),
        pagination=RekuestOffsetPaginationInput(offset=0, limit=len(action_ids)),
    )
    return actions


# -------------------------------------------------------------------------
# 2. Simplifed Structure Extraction & Serialization
# -------------------------------------------------------------------------
async def extract_structure_inputs_for_port(port, value: Any) -> List[StructureInput]:
    if value is None:
        return []

    if port.kind in (PortKind.STRUCTURE, PortKind.MEMORY_STRUCTURE) and port.identifier:
        val_str = str(getattr(value, "id", value))
        if port.kind == PortKind.STRUCTURE:
            registry = get_default_structure_registry()
            val_str = str(
                await registry.get_fullfilled_structure(port.identifier).ashrink(value)
            )
        return [StructureInput(identifier=port.identifier, object=val_str)]

    if port.kind == PortKind.LIST and port.children:
        structures = []
        for item in value:
            structures.extend(
                await extract_structure_inputs_for_port(port.children[0], item)
            )
        return structures
    return []


async def extract_output_structures(action, result: Any) -> List[StructureInput]:
    if not getattr(action, "returns", None):
        return []
    values = result if isinstance(result, tuple) else (result,)

    attached = []
    for port, value in zip(action.returns, values):
        attached.extend(await extract_structure_inputs_for_port(port, value))

    return list({(s.identifier, s.object): s for s in attached}.values())


def serialize_tool_argument(port, value: Any) -> Any:
    if value is None:
        return None

    if port.kind in (PortKind.STRUCTURE, PortKind.MEMORY_STRUCTURE):
        obj_val = (
            value.get("object")
            if isinstance(value, dict) and "object" in value
            else getattr(value, "id", value)
        )
        return {"__identifier": port.identifier, "object": str(obj_val)}

    if port.kind == PortKind.LIST and port.children:
        return [serialize_tool_argument(port.children[0], item) for item in value]

    if (
        port.kind in (PortKind.DICT, PortKind.MODEL)
        and port.children
        and isinstance(value, dict)
    ):
        return {
            child.key: serialize_tool_argument(child, value[child.key])
            for child in port.children
            if child.key in value
        }

    return value


# -------------------------------------------------------------------------
# 3. Tool Actions & LLM Loop
# -------------------------------------------------------------------------
async def search_arkitekt_actions(
    query: str, identifier: str | None = None, limit: int = 5, progress_pct: int = 40
):
    safe_limit = max(1, min(limit, 10))
    await aprogress(progress_pct, f"Searching Arkitekt for '{query}'...")

    if identifier:
        primary = await aprimary_actions(
            identifier=identifier,
            search=query,
            pagination=RekuestOffsetPaginationInput(limit=safe_limit, offset=0),
        )
        actions = await get_actions_by_ids([str(a.id) for a in primary])
    else:
        actions = list(
            await alist_actions(
                filters=ActionFilter(search=query),
                pagination=RekuestOffsetPaginationInput(limit=safe_limit, offset=0),
            )
        )

    if not actions:
        return f"No Arkitekt actions found for query '{query}'.", []

    res = f"Found {len(actions)} actions:\n" + "\n".join(
        f"- {a.name} (id: {a.id})" for a in actions
    )
    return res, actions


async def execute_arkitekt_action(
    action_id: str,
    arguments: dict[str, Any],
    image: Image | None,
    progress_pct: int = 50,
):
    action = await afind(id=action_id)
    await aprogress(progress_pct, f"Executing action '{action.name}'...")

    call_kwargs = dict(arguments)
    for port in action.args:
        if port.key not in call_kwargs and is_autofillable_port(port, image):
            call_kwargs[port.key] = image

    try:
        serialized = {
            p.key: serialize_tool_argument(p, call_kwargs[p.key])
            for p in action.args
            if p.key in call_kwargs
        }
        raw_result = await acall_raw(action=action, kwargs=serialized)
        expanded = await aexpand_returns(
            action, raw_result, structure_registry=get_default_structure_registry()
        )
        result = expanded[0] if len(expanded) == 1 else expanded
    except Exception as error:
        logger.error(f"Action failed: {error}")
        return f"Action '{action.name}' failed: {error}", []

    output_structures = await extract_output_structures(action, result)
    return (
        f"Executed {action.name}. Result:\n{json.dumps(normalize_value(result), default=str)}",
        output_structures,
    )


async def chat_with_tools(
    messages: List[ChatMessageInput],
    model: LLMModel,
    image: Image | None,
    available_action_ids: List[str],
):
    conversation, current_action_ids, pending_structures = (
        list(messages),
        list(available_action_ids),
        [],
    )

    for attempt in range(MAX_TOOL_ROUNDS):
        # Calculate a rough percentage between 10% and 80% based on the attempt round
        pct = min(80, 10 + (attempt * 12))
        await aprogress(pct, f"Thinking (Round {attempt + 1}/{MAX_TOOL_ROUNDS})...")

        actions = await get_actions_by_ids(current_action_ids)
        action_tool_map = {action_tool_name(a): str(a.id) for a in actions}

        answer = await achat(
            model=model,
            messages=conversation,
            tools=(SEARCH_ACTION_TOOL, *[action_to_tool(a, image) for a in actions]),
        )
        msg = answer.choices[0].message

        if not msg.tool_calls:
            return (
                msg.content or "No response generated.",
                current_action_ids,
                pending_structures,
            )

        conversation.append(
            ChatMessageInput(
                role=msg.role,
                content=msg.content,
                tool_calls=[
                    ToolCallInput(
                        id=tc.id,
                        type=tc.type,
                        function=FunctionCallInput(
                            name=tc.function.name, arguments=tc.function.arguments
                        ),
                    )
                    for tc in msg.tool_calls
                ],
            )
        )

        for tool_call in msg.tool_calls:
            name, args = (
                tool_call.function.name,
                json.loads(tool_call.function.arguments or "{}"),
            )
            await aprogress(pct + 5, f"Calling tool: {name}")

            if name == SEARCH_ACTION_TOOL_NAME:
                res, new_actions = await search_arkitekt_actions(
                    args.get("query", ""),
                    args.get("identifier"),
                    args.get("limit", 5),
                    pct + 5,
                )
                for a in new_actions:
                    if str(a.id) not in current_action_ids:
                        current_action_ids.append(str(a.id))
            elif name in action_tool_map:
                res, new_structs = await execute_arkitekt_action(
                    action_tool_map[name], args, image, pct + 5
                )
                pending_structures.extend(new_structs)
            else:
                res = f"Unknown tool: {name}"

            conversation.append(
                ChatMessageInput(
                    role=Role.TOOL, name=name, tool_call_id=tool_call.id, content=res
                )
            )

    await aprogress(85, "Synthesizing final answer...")
    conversation.append(
        ChatMessageInput(
            role=Role.SYSTEM,
            content="Stop calling tools and answer based on gathered info.",
        )
    )
    final_answer = await achat(
        model=model, messages=conversation
    )  # TODO: call local llm

    return (
        final_answer.choices[0].message.content or "Reached execution limit.",
        current_action_ids,
        pending_structures,
    )


# -------------------------------------------------------------------------
# 4. Graph Nodes & Trigger
# -------------------------------------------------------------------------
async def call_llm(state: AgentState):
    resp, action_ids, structs = await chat_with_tools(
        state["messages"],
        state["model"],
        state.get("image"),
        state["available_action_ids"],
    )
    return {
        "latest_response": resp,
        "available_action_ids": action_ids,
        "pending_structures": structs,
    }


async def process_and_send(state: AgentState):
    await aprogress(95, "Sending response...")
    cleartext = re.sub(
        r"\<think\>.*\<\/think\>", "", state["latest_response"], flags=re.DOTALL
    ).strip()
    sent_msg = await asend(
        room=state["room"],
        text=cleartext,
        agent_id="jhnnsrs",
        attach_structures=state["pending_structures"] or None,
    )
    return {
        "messages": [ChatMessageInput(role=Role.ASSISTANT, content=cleartext)],
        "pending_structures": [],
        "sent_message": sent_msg,
    }


class LocalModel(str, Enum):
    GPT_3_5 = "gpt-3.5"
    GPT_4 = "gpt-4"


@register(name="Kaller")
async def reply_to_message(
    message: Message,
    model: Annotated[LLMModel, DescriptionAddin("The LLM model to use")],
) -> Message:

    await aprogress(5, "Initializing assistant...")
    image = None
    if message.attached_structures:
        for struct in message.attached_structures:
            if struct.identifier == "@mikro/image":
                image = await aget_image(id=struct.object)
                break

    workflow = StateGraph(AgentState)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("process_and_send", process_and_send)
    workflow.add_edge(START, "call_llm")
    workflow.add_edge("call_llm", "process_and_send")
    workflow.add_edge("process_and_send", END)
    agent_app = workflow.compile()

    messages = [ChatMessageInput(role=Role.SYSTEM, content=SYSTEM_PROMPT)]
    for m in message.before or []:
        text = (
            f"{m.text}\n[Structures: {', '.join(f'{s.identifier}:{s.object}' for s in m.attached_structures)}]"
            if m.attached_structures
            else m.text
        )
        messages.append(
            ChatMessageInput(
                role=Role.ASSISTANT if str(m.agent.id) == "jhnnsrs" else Role.USER,
                content=text,
            )
        )

    curr_text = message.text
    if message.attached_structures:
        curr_text += f"\n[Structures: {', '.join(f'{s.identifier}:{s.object}' for s in message.attached_structures)}]"
        if image:
            curr_text += f"\n\nAttached image context:\n- image id: {image.id}\n- image name: {image.name}"

    messages.append(ChatMessageInput(role=Role.USER, content=curr_text))

    result_state = await agent_app.ainvoke(
        {
            "room": message.room,
            "model": model,
            "image": image,
            "available_action_ids": [],
            "pending_structures": [],
            "messages": messages,
            "latest_response": "",
            "sent_message": None,
        }
    )

    await aprogress(100, "Done.")
    return result_state["sent_message"]
