import json
import re
import logging
from typing import Any, List, TypedDict, Annotated

import graphql
from enum import Enum
from arkitekt_next import register, easy, aprogress
from arkitekt_next.service_registry import get_default_service_registry
from mikro_next.rath import current_mikro_next_rath
from rekuest_next.rath import current_rekuest_next_rath
from alpaka.rath import current_alpaka_rath
from elektro.rath import current_elektro_rath
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from rekuest_next import Description
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
INSPECT_SCHEMA_TOOL_NAME = "inspect_service_schema"
RUN_QUERY_TOOL_NAME = "run_graphql_query"
ACTION_TOOL_PREFIX = "arkitekt_action"
MAX_TOOL_ROUNDS = 6
MAX_QUERY_RESULT_CHARS = 6000  # truncate large GraphQL results for the LLM

# Maps an Arkitekt service name to a getter for its currently-active rath client.
# The contextvar is set when the app/service context is entered (which it is while
# Kaller runs). Extensible: kabinet/elektro/fluss/unlok expose analogous contextvars.
SERVICE_RATH_GETTERS = {
    "mikro": current_mikro_next_rath.get,
    "rekuest": current_rekuest_next_rath.get,
    "alpaka": current_alpaka_rath.get,
    "elektro": current_elektro_rath.get,
}

# Cache of built graphql schemas keyed by service name.
_SERVICE_SCHEMA_CACHE: dict[str, graphql.GraphQLSchema] = {}

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
INSPECT_SCHEMA_TOOL = ToolInput(
    type=ToolType.FUNCTION,
    function=FunctionDefinitionInput(
        name=INSPECT_SCHEMA_TOOL_NAME,
        description=(
            "Inspect a backend service's GraphQL schema to learn what you can query. "
            "Identify the service either with `service` (e.g. 'mikro', 'rekuest', "
            "'alpaka') or with a structure `identifier` (e.g. '@mikro/image', whose "
            "'@mikro' prefix selects the service). Call WITHOUT `type_name` to get the "
            "Query root fields plus a list of all available type names; then call again "
            "WITH `type_name` (e.g. 'Image') to see that type's fields. Use this before "
            "running a query so you know the exact fields and arguments."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name, e.g. 'mikro', 'rekuest', 'alpaka'.",
                },
                "identifier": {
                    "type": "string",
                    "description": "Structure identifier, e.g. '@mikro/image'. Its prefix selects the service.",
                },
                "type_name": {
                    "type": "string",
                    "description": "A GraphQL type to inspect, e.g. 'Image'. Omit to list Query fields and all type names.",
                },
            },
            "required": [],
        },
    ),
)

RUN_QUERY_TOOL = ToolInput(
    type=ToolType.FUNCTION,
    function=FunctionDefinitionInput(
        name=RUN_QUERY_TOOL_NAME,
        description=(
            "Run a READ-ONLY GraphQL query against a backend service to fetch details. "
            "Mutations and subscriptions are rejected. Identify the service with "
            "`service` or a structure `identifier` (e.g. '@mikro/image' selects 'mikro'). "
            "When fetching details about an attached structure, the structure's 'object' "
            "value is the id to filter on. Inspect the schema first with "
            f"{INSPECT_SCHEMA_TOOL_NAME} so the query uses valid fields."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name, e.g. 'mikro'. Optional if `identifier` is given.",
                },
                "identifier": {
                    "type": "string",
                    "description": "Structure identifier, e.g. '@mikro/image'. Its prefix selects the service.",
                },
                "query": {
                    "type": "string",
                    "description": "The GraphQL query document (read-only). May include variables.",
                },
                "variables": {
                    "type": "object",
                    "description": "Optional variables object for the query.",
                },
            },
            "required": ["query"],
        },
    ),
)

TOOLS = (SEARCH_ACTION_TOOL, INSPECT_SCHEMA_TOOL, RUN_QUERY_TOOL)

SYSTEM_PROMPT = (
    "You are a helpful assistant for Arkitekt and microscopy workflows. "
    "Give direct, useful answers, ask a clarifying question when ambiguous, and use attached images. "
    "Be specific: mention action names, args used, and return types. "
    "Use search_arkitekt_actions to find solutions. For image tasks, prefer @mikro/image. "
    "Attached structures carry an identifier like '@mikro/image' whose '@<service>' prefix "
    "names the backend service, and an 'object' value which is the id. To get more detail "
    "about a structure, first call inspect_service_schema (pass the identifier or service, "
    "then drill into the relevant type), then call run_graphql_query with a read-only query "
    "filtering by that id. "
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
def resolve_service_name(
    service: str | None, identifier: str | None
) -> tuple[str | None, str | None]:
    """Resolve a registered service name from an explicit name or a structure identifier.

    Returns (service_name, error). Exactly one is non-None.
    """
    name = (service or "").strip() or None

    if name is None and identifier:
        # Prefer the structure registry's ward (explicit service tie), fall back to
        # the '@<service>/...' identifier prefix convention.
        try:
            fullfilled = get_default_structure_registry().get_fullfilled_structure(
                identifier
            )
            name = getattr(fullfilled.default_widget, "ward", None)
        except (KeyError, AttributeError):
            name = None
        if not name:
            name = identifier.lstrip("@").split("/")[0] or None

    if not name:
        return None, "Provide a `service` name or a structure `identifier`."

    registry = get_default_service_registry()
    if registry.get(name) is None:
        available = ", ".join(sorted(registry.service_builders.keys())) or "none"
        return None, f"Unknown service '{name}'. Available services: {available}."

    return name, None


def _build_service_schema(service_name: str) -> graphql.GraphQLSchema:
    """Build (and cache) the graphql schema for a service from its bundled SDL."""
    if service_name not in _SERVICE_SCHEMA_CACHE:
        sdl = get_default_service_registry().get(service_name).get_graphql_schema()
        # assume_valid bypasses unknown directives like @oneOf in the bundled SDL.
        _SERVICE_SCHEMA_CACHE[service_name] = graphql.build_schema(
            sdl, assume_valid=True
        )
    return _SERVICE_SCHEMA_CACHE[service_name]


def _is_public_type(name: str) -> bool:
    return not name.startswith("__") and name not in ("_Entity", "_Service", "_Any")


async def inspect_service_schema(
    service: str | None = None,
    identifier: str | None = None,
    type_name: str | None = None,
) -> str:
    name, error = resolve_service_name(service, identifier)
    if error:
        return error

    try:
        schema = _build_service_schema(name)
    except Exception as err:  # pragma: no cover - defensive
        logger.error(f"Failed to build schema for '{name}': {err}")
        return f"Could not load schema for service '{name}': {err}"

    if type_name:
        gql_type = schema.type_map.get(type_name)
        if gql_type is None:
            candidates = [
                t
                for t in schema.type_map
                if _is_public_type(t) and type_name.lower() in t.lower()
            ]
            hint = (
                f" Did you mean: {', '.join(sorted(candidates)[:10])}?"
                if candidates
                else ""
            )
            return f"Type '{type_name}' not found in service '{name}'.{hint}"
        return graphql.print_type(gql_type)

    parts = []
    if schema.query_type is not None:
        parts.append(graphql.print_type(schema.query_type))
    type_names = sorted(t for t in schema.type_map if _is_public_type(t))
    parts.append(
        "Available types (call inspect_service_schema with a `type_name` to expand):\n"
        + ", ".join(type_names)
    )
    return f"Schema for service '{name}':\n\n" + "\n\n".join(parts)


async def run_graphql_query(
    service: str | None = None,
    identifier: str | None = None,
    query: str = "",
    variables: dict[str, Any] | None = None,
    progress_pct: int = 50,
) -> str:
    name, error = resolve_service_name(service, identifier)
    if error:
        return error

    if not query.strip():
        return "Provide a GraphQL `query` string."

    try:
        document = graphql.parse(query)
    except graphql.GraphQLSyntaxError as err:
        return f"GraphQL syntax error: {err}"

    for definition in document.definitions:
        operation = getattr(definition, "operation", None)
        if operation is not None and operation != graphql.OperationType.QUERY:
            return (
                "This tool is read-only; mutations/subscriptions are not allowed."
            )

    getter = SERVICE_RATH_GETTERS.get(name)
    try:
        # mikro's contextvar has no default and raises LookupError when unset.
        rath = getter() if getter else None
    except LookupError:
        rath = None
    if rath is None:
        return (
            f"No active GraphQL client for service '{name}'. "
            f"Queryable services: {', '.join(sorted(SERVICE_RATH_GETTERS))}."
        )

    await aprogress(progress_pct, f"Querying {name} GraphQL...")
    try:
        result = await rath.aquery(query, variables or {})
    except Exception as err:
        logger.error(f"GraphQL query on '{name}' failed: {err}")
        return f"GraphQL query failed: {err}"

    payload = json.dumps(normalize_value(result.data), default=str)
    if len(payload) > MAX_QUERY_RESULT_CHARS:
        payload = payload[:MAX_QUERY_RESULT_CHARS] + "...(truncated)"
    return payload


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
            tools=(
                SEARCH_ACTION_TOOL,
                INSPECT_SCHEMA_TOOL,
                RUN_QUERY_TOOL,
                *[action_to_tool(a, image) for a in actions],
            ),
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
            elif name == INSPECT_SCHEMA_TOOL_NAME:
                res = await inspect_service_schema(
                    args.get("service"),
                    args.get("identifier"),
                    args.get("type_name"),
                )
            elif name == RUN_QUERY_TOOL_NAME:
                res = await run_graphql_query(
                    args.get("service"),
                    args.get("identifier"),
                    args.get("query", ""),
                    args.get("variables"),
                    pct + 5,
                )
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
    model: Annotated[LLMModel, Description("The LLM model to use")],
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
