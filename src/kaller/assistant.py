"""The assistant's turn: find an installed action, call it as a held child, read what came back.

Everything here takes its clients as parameters, so the tests hand in fakes; `app.py` wires
the real ones to a room.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from alpaka import Alpaka
from alpaka.api.schema import MemoryKind, Message, StructureInput
from arkitekt import (
    AssignPrompt,
    BlokLocalInput,
    DeniedError,
    DismissedError,
    PortKind,
    Task,
    TaskBlok,
)
from arkitekt_spec.declare.errors import BlokRefused
from kabinet import Kabinet
from kabinet.api.schema import InstallableDefinition
from rath.operation import GraphQLException
from rekuest.client.client import Rekuest
from rekuest.protocol.schema import (
    ActionFilter,
    DemandKind,
    ImplementationActionFilter,
    ImplementationFilter,
    OffsetPaginationInput,
    PortDemandInput,
    PortMatchInput,
)

from kaller.schema_engine import (
    GRAPHQL_TOOLS,
    GraphQLEngine,
    brief,
    clip,
    graphql_tools,
    run_graphql_tool,
)

logger = logging.getLogger(__name__)

#: The agent name this app writes under: its own messages in the history.
AGENT_ID = "kaller"
#: What a room replyer takes; actions taking one are never offered.
MESSAGE = "@alpaka/message"

FIND_ACTIONS_TOOL = "find_actions"
CALL_ACTION_TOOL = "call_action"
READ_STATE_TOOL = "read_state"
FIND_INSTALLABLE_TOOL = "find_installable"
INSTALL_TOOL = "install"
REMEMBER_TOOL = "remember"
RECALL_TOOL = "recall"
FORGET_TOOL = "forget"
ASK_USER_TOOL = "ask_user"

#: What the user answers a question in: the question, a text field, a send button.
ASK_BLOK = """<Html>
  <Card>
    <Label>{args.question}</Label>
    <Input bind={answer} />
    <Button onClick={task.answer(values: {answer: answer})}>Send</Button>
  </Card>
</Html>"""
#: How many of the assigner's memories the index a turn starts with lists, newest first.
MAX_MEMORY_INDEX = 200
#: How many memories one recall reads whole.
MAX_RECALLED = 5

#: What an installer takes, the definition it makes runnable, and what it returns: the pod it runs in.
DEFINITION = "@kabinet/definition"
POD = "@kabinet/pod"
#: How long an install may take to come online (pull, start, log in, register), and how
#: often it is looked at meanwhile.
INSTALL_TIMEOUT_SECONDS = 600
INSTALL_POLL_SECONDS = 5

#: Rounds before the model is made to answer: search, call, maybe a second
#: search and call after a refusal or a failure, or a store search and an install
#: before the call.
MAX_ROUNDS = 10
#: Non-system messages kept, newest last.
MAX_HISTORY = 24
#: Agents whose state one read_state shows: an action can run on several.
MAX_STATE_AGENTS = 5
#: Changes read_state lists on each side of the moment it reads, by default and at most.
STATE_CHANGES = 5
MAX_STATE_CHANGES = 20

SYSTEM_PROMPT = """\
You run actions for the user. Actions are installed on the platform by apps: segmenting \
images, measuring, converting data, moving hardware. When the user asks for work, find an \
action for it with find_actions and run it with call_action, then say in a few sentences \
what ran and what came back. If nothing fits, say so; if the request is unclear, ask.

Arguments: use the action's own argument names exactly. An argument asking for a structure \
the room carries (shown as [Attached: identifier:id]) is filled in for you: leave it out. \
Give other arguments only when the user said them or the action needs them; otherwise let \
the action's defaults stand.

An action's agent may keep state: where a stage is, what is loaded, what it last did. \
read_state shows it. Read it before a call whose arguments depend on it, and again \
afterwards when the user wants to know what changed. It also reads the past: give it a \
time (at) or seconds_ago, and it shows the state as it was then and the changes around \
it, each with its time; read at a change's time to see the state right after it.

To learn about the data itself -- an attached image's name and size, a room's history, \
which releases an app has -- read the backend it lives in: an attachment \
[Attached: @mikro/image:42] lives in mikro (the @ prefix names the service) under id 42. \
Search that service's schema with graphql_search_schema, describe the types you need with \
graphql_describe, then graphql_query for exactly that id. Queries only read; to change \
anything, run an action.

Give call_action a prompt the user reads before they approve the call: a one-line title \
saying what will run on what ("Segment nuclei in plate 2"), and a description of a sentence or \
two saying why. The server shows the action, its agent and the arguments next to it, so do not \
repeat them; explain. If the user declines, accept it, do not call the \
same thing again, and say what you could do instead.

If no installed action fits, look in the app store with find_installable. When something \
there fits, offer it, or install it with install (giving it a prompt the same way) when the \
user asked for the work to be done; it comes back as an action you can call_action right \
away. Installing may take minutes and may ask the user to approve the app first.

You keep memories of the user you work for; their index follows. When they tell you \
something lasting -- how they want results, what they work on, a correction -- remember \
it (kind USER, FEEDBACK, PROJECT or REFERENCE), under the name of the memory it updates \
if one is about the same thing. Recall a memory before relying on it. Never remember what \
only matters to this conversation. When something only the user can tell you is missing \
and not remembered, ask_user; offer to remember the answer if it will matter again.\
"""


# ---------------------------------------------------------------------------
# Helpers shared with chat.py's shape
# ---------------------------------------------------------------------------


def keep_recent(messages: list[dict]) -> list[dict]:
    """Trim to the last ``MAX_HISTORY`` non-system messages, system messages pinned.

    A tool result whose call was trimmed away goes too: the provider refuses a
    tool message without its ``tool_calls``.
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"][-MAX_HISTORY:]
    while rest and rest[0].get("role") == "tool":
        rest.pop(0)
    return system + rest


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def tool_result(call: dict, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": name,
        "content": content,
    }


async def _recorded(
    task: Task | None, fn: Callable[[], Any], key: str | None = None
) -> Any:
    """``fn()`` (awaited), recorded by the task when there is one: a resumed turn gets it back."""
    if task is not None:
        return await task.arecord(fn, key=key)
    value = fn()
    return await value if inspect.isawaitable(value) else value


async def stream_completion(
    client: Any,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    on_token: Any,
) -> tuple[str, list[dict]]:
    """One streamed completion: its text, and the tool calls accumulated by ``index``."""
    content: list[str] = []
    partial: dict[int, dict] = {}

    stream = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None, stream=True
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content.append(delta.content)
            await on_token(delta.content)
        for fragment in delta.tool_calls or []:
            call = partial.setdefault(
                fragment.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if fragment.id:
                call["id"] = fragment.id
            if fragment.function and fragment.function.name:
                call["function"]["name"] = fragment.function.name
            if fragment.function and fragment.function.arguments:
                call["function"]["arguments"] += fragment.function.arguments

    return "".join(content), [partial[index] for index in sorted(partial)]


# ---------------------------------------------------------------------------
# Finding actions
# ---------------------------------------------------------------------------


def _value(enum_or_str: Any) -> str:
    return str(getattr(enum_or_str, "value", enum_or_str))


def render_port(port: Any) -> str:
    line = f"{port.key}: {_value(port.kind)}"
    if getattr(port, "identifier", None):
        line += f" {port.identifier}"
    if getattr(port, "nullable", False):
        line += " (optional)"
    if getattr(port, "description", None):
        line += f"  # {brief(port.description, 80)}"
    return line


def render_action(action: Any) -> str:
    args = "\n".join(f"      {render_port(port)}" for port in action.args)
    returns = ", ".join(render_port(port) for port in action.returns) or "nothing"
    return (
        f"- {action.name} (id: {action.id})"
        + (f": {brief(action.description)}" if action.description else "")
        + f"\n    arguments:\n{args or '      (none)'}"
        + f"\n    returns: {returns}"
    )


def is_replyer(action: Any) -> bool:
    """Whether the action answers a room message: another assistant, or this one."""
    return any(getattr(port, "identifier", None) == MESSAGE for port in action.args)


async def find_actions(rekuest: Rekuest, query: str, limit: int = 8) -> str:
    """The installed actions matching ``query``: only what a connected agent can run now."""
    limit = max(1, min(limit, 10))
    actions = [
        action
        for action in await rekuest.alist_actions(
            filters=ActionFilter(search=query, assignable=True),
            # Room replyers are dropped after the fetch: ask for a few more.
            pagination=OffsetPaginationInput(limit=limit + 4, offset=0),
        )
        if not is_replyer(action)
    ][:limit]
    if not actions:
        return (
            f"No installed action matches '{query}'. Try other words; if none fits, "
            "the app that would do it is not running."
        )
    return f"{len(actions)} actions:\n" + "\n".join(render_action(a) for a in actions)


# ---------------------------------------------------------------------------
# Reading what an agent is doing
# ---------------------------------------------------------------------------


def moment(at: Any, seconds_ago: Any, now: datetime.datetime) -> datetime.datetime | None:
    """The time to read the state at, None for now. A time without a zone is UTC.

    Raises ValueError, with a message for the model, on a time it cannot use.
    """
    if at:
        try:
            when = datetime.datetime.fromisoformat(str(at))
        except ValueError:
            raise ValueError(f"at is not an ISO time: {at!r}.") from None
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.UTC)
    elif seconds_ago:
        try:
            when = now - datetime.timedelta(seconds=float(seconds_ago))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"seconds_ago is not a number of seconds: {seconds_ago!r}.") from None
    else:
        return None
    if when > now:
        raise ValueError(f"{when.isoformat()} is in the future (it is {now.isoformat()} now).")
    return when


def render_change(patch: Any) -> str:
    """One recorded change: when, what, and the task that made it."""
    by = f"  (task {patch.task.id}: {patch.task.action.name})" if patch.task else ""
    return f"{patch.timestamp.isoformat()}  {patch.op} {patch.interface}{patch.path}{by}"


async def read_state(
    rekuest: Rekuest,
    action_id: str,
    at: Any = None,
    seconds_ago: Any = None,
    changes: Any = STATE_CHANGES,
    now: datetime.datetime | None = None,
) -> str:
    """The state of every agent that can run ``action_id`` now, by state interface.

    At ``at`` (or ``seconds_ago``), the state as it was then; with the changes recorded
    just before that moment, and after it when it is in the past.
    """
    now = now or datetime.datetime.now(datetime.UTC)
    try:
        when = moment(at, seconds_ago, now)
        count = max(0, min(int(changes if changes is not None else STATE_CHANGES), MAX_STATE_CHANGES))
    except (TypeError, ValueError) as error:
        return str(error)

    implementations = await rekuest.alist_implementations(
        filters=ImplementationFilter(action=ImplementationActionFilter(ids=[action_id]), active=True),
        pagination=OffsetPaginationInput(limit=MAX_STATE_AGENTS, offset=0),
    )
    agents = list(dict.fromkeys(implementation.agent.id for implementation in implementations))
    if not agents:
        return f"No connected agent runs action {action_id!r}."

    lines = [f"It is {now.isoformat()} now. " + (f"State at {when.isoformat()}:" if when else "Current state:")]
    for agent_id in agents:
        try:
            checked = await rekuest.acheckout_agent(
                agent=agent_id,
                timestamp=when,
                backward_patch_count=count,
                forward_patch_count=count if when else 0,
            )
        except GraphQLException as error:
            lines.append(f"- agent {agent_id}: its state could not be read: {error.message}")
            continue
        if not checked.values:
            lines.append(f"- agent {agent_id}: keeps no state")
            continue
        lines.append(f"- agent {agent_id}:")
        lines += [
            f"    {interface}: {json.dumps(value, default=str)}" for interface, value in checked.values.items()
        ]
        if checked.backward_patches:
            lines.append("  changes up to then, oldest first:" if when else "  last changes, oldest first:")
            lines += [f"    {render_change(patch)}" for patch in checked.backward_patches]
        if checked.forward_patches:
            lines.append("  changes after, oldest first:")
            lines += [f"    {render_change(patch)}" for patch in checked.forward_patches]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calling one
# ---------------------------------------------------------------------------


def shrink_argument(port: Any, value: Any) -> Any:
    """Turn a model-supplied argument into the wire form the port expects: a structure as
    the reference ``{"__identifier", "object"}`` the server checks arguments for (a bare id
    is only a default's form)."""
    if value is None:
        return None
    if port.kind in (PortKind.STRUCTURE, PortKind.MEMORY_STRUCTURE, PortKind.INTERFACE):
        if isinstance(value, dict) and "object" in value:
            identifier = value.get("__identifier") or port.identifier
            return {"__identifier": identifier, "object": str(value["object"])}
        return {"__identifier": port.identifier, "object": str(getattr(value, "id", value))}
    if port.kind == PortKind.LIST and port.children:
        return [shrink_argument(port.children[0], item) for item in value]
    if port.kind in (PortKind.DICT, PortKind.MODEL) and port.children and isinstance(value, dict):
        return {
            child.key: shrink_argument(child, value[child.key])
            for child in port.children
            if child.key in value
        }
    return value


def structures_returned(port: Any, value: Any) -> list[tuple[str, int]]:
    """The (identifier, id) pairs a return port carried, for the reply to attach."""
    if value is None:
        return []
    if port.kind == PortKind.STRUCTURE and port.identifier and str(value).isdigit():
        return [(port.identifier, int(value))]
    if port.kind == PortKind.LIST and port.children and isinstance(value, list):
        return [pair for item in value for pair in structures_returned(port.children[0], item)]
    return []


def render_return(port: Any, value: Any) -> str:
    if port.kind == PortKind.STRUCTURE and port.identifier:
        return f"{port.key}: {port.identifier} {value}"
    return f"{port.key}: {json.dumps(value, default=str)}"


def prompt_of(prompt: Any, fallback_title: str) -> AssignPrompt:
    """The model's prompt as an ``AssignPrompt``; ``fallback_title`` when it gives none.

    A bare string (some models flatten the object) is the description.
    """
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError:
            prompt = {"description": prompt}
    if not isinstance(prompt, dict):
        prompt = {}
    title = " ".join(str(prompt.get("title") or "").split())
    description = str(prompt.get("description") or "").strip()
    return AssignPrompt(title=title or fallback_title, description=description or None)


@dataclass
class Outcome:
    """What a call came to: text for the model, and what the reply attaches."""

    text: str
    attachments: list[tuple[str, int]] = field(default_factory=list)


async def call_action(
    rekuest: Rekuest,
    task: Task | None,
    action_id: str,
    arguments: Any,
    prompt: Any,
    context: dict[str, str],
    call_key: str,
) -> Outcome:
    """Call an action as this turn's child, once the user approves it (or at once in auto-mode).

    Every outcome is text for the model; nothing raises.
    """
    if task is None:
        return Outcome("Nothing can be called here: this turn runs without a task.")
    if isinstance(arguments, str):
        # Some models send the object as a JSON string.
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as error:
            return Outcome(f"arguments is not JSON: {error}")
    if not isinstance(arguments, dict):
        return Outcome("arguments must be an object of argument name -> value.")
    try:
        action = await rekuest.afind(id=action_id)
    except GraphQLException as error:
        return Outcome(f"There is no action {action_id!r}: {error.message}")
    if is_replyer(action):
        return Outcome(f"{action.name} answers room messages; it is not something to run here.")

    call_kwargs = dict(arguments)
    for port in action.args:
        # What the room carried fills an argument asking for it: the model need not copy ids.
        identifier = getattr(port, "identifier", None)
        if port.key not in call_kwargs and identifier in context:
            call_kwargs[port.key] = context[identifier]
    unknown = sorted(set(call_kwargs) - {port.key for port in action.args})
    if unknown:
        return Outcome(
            f"{action.name} takes no argument {', '.join(unknown)}. "
            f"Its arguments: {', '.join(port.key for port in action.args) or 'none'}."
        )
    missing = [
        render_port(port)
        for port in action.args
        if port.key not in call_kwargs and not port.nullable and port.default is None
    ]
    if missing:
        return Outcome(
            f"{action.name} needs: {'; '.join(missing)}. Nothing attached in the room "
            "fills them: ask the user, or pass them."
        )
    shrunk = {
        port.key: shrink_argument(port, call_kwargs[port.key])
        for port in action.args
        if port.key in call_kwargs
    }

    prompt = prompt_of(prompt, f"Run {action.name}")
    try:
        raw = await task.acall_raw(action=action, kwargs=shrunk, prompt=prompt, call_key=call_key)
    except DeniedError as denied:
        said = f' They said: "{denied.answer}"' if denied.answer else ""
        return Outcome(f"The user declined to let {action.name} run; nothing ran.{said}")
    except Exception as error:  # noqa: BLE001 - any failure is text for the model
        logger.warning("Action %s failed: %s", action_id, error)
        return Outcome(f"{action.name} failed: {error}")

    raw = raw or {}
    returned = [port for port in action.returns if port.key in raw]
    attachments = [
        pair for port in returned for pair in structures_returned(port, raw[port.key])
    ]
    lines = [render_return(port, raw[port.key]) for port in returned]
    return Outcome(
        f"{action.name} ran. "
        + ("It returned:\n" + "\n".join(lines) if lines else "It returned nothing."),
        attachments,
    )


# ---------------------------------------------------------------------------
# Installing what is not
# ---------------------------------------------------------------------------


def render_installable(definition: InstallableDefinition) -> str:
    releases = {
        f.release.id: f"{f.release.app.identifier} {f.release.version} (release {f.release.id})"
        for f in definition.flavours
    }
    return (
        f"- {definition.name} (definition id: {definition.id})"
        + (f": {brief(definition.description)}" if definition.description else "")
        + f"\n    provided by: {', '.join(releases.values()) or 'no release'}"
    )


async def find_installable(kabinet: Kabinet, query: str, limit: int = 8) -> str:
    """What the app store could install to do ``query``."""
    found = await kabinet.asearch_installable(query, limit=max(1, min(limit, 10)))
    if not found:
        return f"Nothing in the app store matches '{query}'."
    return f"{len(found)} installable:\n" + "\n".join(render_installable(d) for d in found)


async def find_installer(rekuest: Rekuest) -> Any | None:
    """An installer that is running: an action taking a definition and returning the pod it runs in."""
    installers = await rekuest.alist_actions(
        filters=ActionFilter(
            demands=[
                PortDemandInput(
                    kind=DemandKind.ARGS,
                    matches=[PortMatchInput(kind=PortKind.STRUCTURE, identifier=DEFINITION)],
                    force_length=1,
                ),
                PortDemandInput(
                    kind=DemandKind.RETURNS,
                    matches=[PortMatchInput(kind=PortKind.STRUCTURE, identifier=POD)],
                    force_length=1,
                ),
            ],
            assignable=True,
        ),
        pagination=OffsetPaginationInput(limit=1, offset=0),
    )
    return installers[0] if installers else None


async def runnable_action(rekuest: Rekuest, action_hash: str) -> Any | None:
    """The action with ``action_hash``, when an agent can take it now; else None."""
    try:
        action = await rekuest.afind(hash=action_hash)
    except GraphQLException:
        return None  # not registered yet
    running = await rekuest.alist_actions(
        filters=ActionFilter(ids=[action.id], assignable=True),
        pagination=OffsetPaginationInput(limit=1, offset=0),
    )
    return running[0] if running else None


async def wait_until_runnable(rekuest: Rekuest, action_hash: str, report: Callable[[int, str], Any] | None) -> Any | None:
    """Poll until the installed action can run, or the install timeout passes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + INSTALL_TIMEOUT_SECONDS
    while True:
        action = await runnable_action(rekuest, action_hash)
        if action is not None or loop.time() >= deadline:
            return action
        if report is not None:
            await report(90, "Waiting for the installed app to come online...")
        await asyncio.sleep(INSTALL_POLL_SECONDS)


async def install(
    rekuest: Rekuest,
    kabinet: Kabinet,
    task: Task | None,
    definition_id: str,
    prompt: Any,
    call_key: str,
    report: Callable[[int, str], Any] | None = None,
) -> Outcome:
    """Install what provides ``definition_id``, and wait until its action can run.

    The installer runs as this turn's held child; if the release needs approving, the
    installer asks the user through the app they called this from. Nothing raises.
    """
    if task is None:
        return Outcome("Nothing can be installed here: this turn runs without a task.")
    try:
        definition = await kabinet.aget_installable(definition_id)
    except GraphQLException as error:
        return Outcome(f"There is no definition {definition_id!r} in the app store: {error.message}")
    # Its releases, attached when it fails: the user can open them and act themselves.
    releases = [("@kabinet/release", int(f.release.id)) for f in definition.flavours if str(f.release.id).isdigit()][:1]

    already = await runnable_action(rekuest, definition.hash)
    if already is not None:
        return Outcome(f"{definition.name} is already installed and running:\n{render_action(already)}")

    installer = await find_installer(rekuest)
    if installer is None:
        return Outcome(
            f"No installer is running, so {definition.name} cannot be installed now. "
            "Tell the user to install it from the app store page.",
            releases,
        )

    try:
        raw = await task.acall_raw(
            action=installer,
            kwargs={port.key: definition_id for port in installer.args[:1]},
            prompt=prompt_of(prompt, f"Install {definition.name}"),
            call_key=call_key,
        )
    except DeniedError as denied:
        said = f' They said: "{denied.answer}"' if denied.answer else ""
        return Outcome(f"The user declined to install {definition.name}; nothing was installed.{said}")
    except Exception as error:  # noqa: BLE001 - any failure is text for the model
        logger.warning("Installing %s failed: %s", definition_id, error)
        return Outcome(f"Installing {definition.name} failed: {error}", releases)

    raw = raw or {}
    pods = [pair for port in installer.returns if port.key in raw for pair in structures_returned(port, raw[port.key])]

    async def wait() -> dict[str, Any] | None:
        action = await wait_until_runnable(rekuest, definition.hash, report)
        return None if action is None else {"text": render_action(action)}

    ready = await _recorded(task, wait, key=f"wait:{call_key}")
    if ready is None:
        return Outcome(
            f"{definition.name} was installed but has not come online within "
            f"{INSTALL_TIMEOUT_SECONDS // 60} minutes. It may still be starting; look again with find_actions.",
            pods,
        )
    return Outcome(f"{definition.name} is installed and running:\n{ready['text']}", pods)


# ---------------------------------------------------------------------------
# Remembering the user, and asking them
# ---------------------------------------------------------------------------


def render_memory(memory: Any) -> str:
    return f"## {memory.name} ({_value(memory.kind)})\n{memory.description}\n\n{memory.content}"


async def memory_preamble(alpaka: Alpaka) -> str:
    """Who assigned the turn and the index of what is remembered about them, for the system prompt.

    alpaka answers only for the turn's own task: a refusal (no task, another organization)
    leaves the model without memory, told so, rather than failing the turn.
    """
    try:
        assigner = await alpaka.aassigner()
        index = await alpaka.amemory_index(limit=MAX_MEMORY_INDEX)
    except GraphQLException as error:
        return f"Memory is not available in this turn ({error.message}): do not remember or recall."
    who = assigner.name or assigner.preferred_username or assigner.sub
    lines = [f"You work for {who}."]
    if index:
        lines.append("What you remember about them (recall a name to read it whole):")
        lines += [f"- {memory.name} ({_value(memory.kind)}): {memory.description}" for memory in index]
    else:
        lines.append("You remember nothing about them yet.")
    return "\n".join(lines)


async def remember(alpaka: Alpaka, name: Any, description: Any, content: Any, kind: Any) -> str:
    name, description, content = str(name or "").strip(), str(description or "").strip(), str(content or "").strip()
    if not (name and description and content):
        return "A memory needs a name, a one-line description and its content."
    try:
        memory_kind = MemoryKind(str(kind or "USER").upper())
    except ValueError:
        return f"kind is one of {', '.join(k.value for k in MemoryKind)}, not {kind!r}."
    try:
        memory = await alpaka.aremember(name=name, description=description, content=content, kind=memory_kind)
    except GraphQLException as error:
        return f"Could not remember it: {error.message}"
    return f"Remembered '{memory.name}'."


async def recall(alpaka: Alpaka, query: Any = None, name: Any = None) -> str:
    """The memories named, or the ones matching ``query``, whole."""
    try:
        if name:
            memories = await alpaka.arecall(names=[str(name)], limit=1)
        else:
            memories = await alpaka.arecall(search=str(query or ""), limit=MAX_RECALLED)
    except GraphQLException as error:
        return f"Could not recall: {error.message}"
    if not memories:
        return f"Nothing is remembered under '{name}'." if name else f"Nothing remembered matches '{query}'."
    return "\n\n".join(render_memory(memory) for memory in memories)


async def forget(alpaka: Alpaka, name: Any) -> str:
    try:
        forgotten = await alpaka.aforget(name=str(name or ""))
    except GraphQLException as error:
        return f"Could not forget it: {error.message}"
    return f"Forgot '{name}'." if forgotten else f"Nothing was remembered under '{name}'."


async def ask_user(task: Task | None, question: Any) -> str:
    """Hold the turn on a question in the room, and hand back the answer.

    ``task.aask`` records the answer, so a turn taken over gets it back without asking twice.
    """
    question = str(question or "").strip()
    if not question:
        return "Ask a question."
    if task is None:
        return "Nobody can be asked here: this turn runs without a task. Ask in your answer instead."
    blok = TaskBlok(
        source=ASK_BLOK,
        name=question,
        locals=[BlokLocalInput(key="answer", default="")],
        returns=[{"key": "answer", "kind": "STRING", "nullable": False, "effects": []}],
    )
    try:
        values = await task.adynamic_ask(blok, args={"question": question})
    except DismissedError:
        return "The user dismissed the question without answering. Go on without it, or ask in words."
    except BlokRefused as refused:
        # The question blok is this app's own BSX: a refusal is a bug here, said plainly.
        logger.error("The question blok was refused: %s", refused.reason)
        return f"The question could not be shown ({refused.reason}). Ask in your answer instead."
    answer = str(values.get("answer") or "").strip()
    if not answer:
        return "The user answered with nothing. Go on without it, or ask in words."
    return f"The user answered: {answer}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    _tool(
        FIND_ACTIONS_TOOL,
        "Search the actions installed on the platform that can run now. Lists each one's "
        "id, arguments and returns.",
        {
            "query": {
                "type": "string",
                "description": "What the action should do, e.g. 'segment nuclei'.",
            },
        },
        ["query"],
    ),
    _tool(
        CALL_ACTION_TOOL,
        "Run an action found with find_actions. Unless auto-mode is on, the user approves "
        "it first; their refusal, or the action's result, comes back.",
        {
            "action_id": {"type": "string", "description": "The action's id."},
            "arguments": {
                "type": "object",
                "description": "The action's arguments by name. An argument asking for "
                "something attached in the room is filled in: omit it.",
            },
            "prompt": {
                "type": "object",
                "description": "What the user reads before approving the call.",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "One line: what will run on what, e.g. 'Segment nuclei in plate 2'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A sentence or two on why.",
                    },
                },
                "required": ["title"],
            },
        },
        ["action_id", "prompt"],
    ),
    _tool(
        READ_STATE_TOOL,
        "Read the state of the agents that run an action: what they keep track of, e.g. where "
        "a stage is. Now by default, or as it was at any earlier time. Lists the changes "
        "recorded around that moment, each with its time and the task that made it.",
        {
            "action_id": {"type": "string", "description": "The action's id, from find_actions."},
            "at": {
                "type": "string",
                "description": "An earlier time to read the state at, ISO 8601, e.g. a change's time. "
                "Leave out for now.",
            },
            "seconds_ago": {
                "type": "number",
                "description": "Instead of at: how many seconds back to read the state.",
            },
            "changes": {
                "type": "integer",
                "description": f"How many changes to list on each side (default {STATE_CHANGES}, "
                f"at most {MAX_STATE_CHANGES}).",
            },
        },
        ["action_id"],
    ),
    _tool(
        FIND_INSTALLABLE_TOOL,
        "Search the app store for what could be installed, when no installed action fits. "
        "Lists each definition's id and the releases providing it.",
        {"query": {"type": "string", "description": "What it should do, e.g. 'segment nuclei'."}},
        ["query"],
    ),
    _tool(
        INSTALL_TOOL,
        "Install what provides a definition found with find_installable, and wait until it can "
        "run. Unless auto-mode is on, the user approves it first; it comes back as an action "
        "for call_action.",
        {
            "definition_id": {"type": "string", "description": "The definition's id."},
            "prompt": {
                "type": "object",
                "description": "What the user reads before approving the install.",
                "properties": {
                    "title": {"type": "string", "description": "One line, e.g. 'Install the nuclei segmenter'."},
                    "description": {"type": "string", "description": "A sentence or two on why."},
                },
                "required": ["title"],
            },
        },
        ["definition_id", "prompt"],
    ),
    _tool(
        REMEMBER_TOOL,
        "Remember something lasting about the user, replacing the memory of that name if there is one.",
        {
            "name": {"type": "string", "description": "A short slug, e.g. 'export-format'. Reuse a name from the index to update it."},
            "description": {"type": "string", "description": "One line saying what the memory holds: what you read in the index."},
            "content": {"type": "string", "description": "The memory itself: the fact, and for feedback why and when it applies."},
            "kind": {"type": "string", "enum": [kind.value for kind in MemoryKind], "description": "USER (who they are), FEEDBACK (how they want work done), PROJECT (what they work on) or REFERENCE (where to look)."},
        },
        ["name", "description", "content", "kind"],
    ),
    _tool(
        RECALL_TOOL,
        "Read memories about the user whole: one by its name from the index, or those matching a query.",
        {
            "name": {"type": "string", "description": "A memory's name from the index."},
            "query": {"type": "string", "description": "Instead of name: what the memory would be about."},
        },
        [],
    ),
    _tool(
        FORGET_TOOL,
        "Forget a memory about the user that turned out wrong or that they asked you to drop.",
        {"name": {"type": "string", "description": "The memory's name."}},
        ["name"],
    ),
    _tool(
        ASK_USER_TOOL,
        "Ask the user a question in the room and wait for their answer. Only for what only they can tell you.",
        {"question": {"type": "string", "description": "One clear question."}},
        ["question"],
    ),
]


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    answer: str
    message_id: str | None
    attachments: list[tuple[str, int]]


def attach_all(writer: Any, attachments: list[tuple[str, int]]) -> None:
    for identifier, object_id in dict.fromkeys(attachments):
        writer.attach(StructureInput(identifier=identifier, object=object_id))


async def converse(
    rekuest: Rekuest,
    client: Any,
    open_room_stream: Callable[[str], Any],
    room: str,
    model: str,
    messages: list[dict],
    context: dict[str, str],
    task: Task | None = None,
    report: Callable[[int, str], Any] | None = None,
    kabinet: Kabinet | None = None,
    alpaka: Alpaka | None = None,
    engines: dict[str, GraphQLEngine] | None = None,
) -> Turn:
    """Think, act, repeat; the final answer carries every structure a call returned.

    Every client is a parameter, so a test can hand in fakes.
    """
    attachments: list[tuple[str, int]] = []
    tools = TOOLS + graphql_tools(engines or {})

    for round_ in range(MAX_ROUNDS + 1):
        last_round = round_ == MAX_ROUNDS
        if report is not None:
            await report(min(80, 10 + round_ * 10), f"Round {round_ + 1}...")

        async def one_round(last_round: bool = last_round) -> dict[str, Any]:
            prompt = keep_recent(messages)
            if last_round:
                prompt.append(
                    {
                        "role": "system",
                        "content": "Stop calling tools and tell the user where things stand.",
                    }
                )
            opened: dict[str, Any] = {"manager": None, "writer": None}

            async def on_token(token: str) -> None:
                if opened["writer"] is None:
                    opened["manager"] = open_room_stream(room)
                    opened["writer"] = await opened["manager"].__aenter__()
                await opened["writer"].append(token)

            try:
                text, tool_calls = await stream_completion(
                    client, model, prompt, None if last_round else tools, on_token
                )
            except BaseException as error:
                if opened["manager"] is not None:
                    await opened["manager"].__aexit__(
                        type(error), error, error.__traceback__
                    )
                raise

            if opened["manager"] is not None:
                if not tool_calls:
                    # The answer carries what it talks about.
                    attach_all(opened["writer"], attachments)
                await opened["manager"].__aexit__(None, None, None)

            return {
                "text": text,
                "tool_calls": tool_calls,
                "message_id": opened["writer"].id if opened["writer"] else None,
            }

        done = await _recorded(task, one_round, key=f"round:{round_}")
        if not done["tool_calls"]:
            answer = strip_thinking(done["text"]) or "No response generated."
            return Turn(answer, done["message_id"], attachments)

        messages.append(
            {
                "role": "assistant",
                "content": done["text"] or None,
                "tool_calls": done["tool_calls"],
            }
        )
        for index, call in enumerate(done["tool_calls"]):
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as error:
                messages.append(tool_result(call, name, f"Could not parse arguments: {error}"))
                continue
            if report is not None:
                await report(90, f"Running {name}...")

            if name == FIND_ACTIONS_TOOL:
                search = partial(find_actions, rekuest, str(arguments.get("query") or ""))
                text = await _recorded(task, search, key=f"find:{round_}:{index}")
            elif name == READ_STATE_TOOL:
                look = partial(
                    read_state,
                    rekuest,
                    str(arguments.get("action_id") or ""),
                    at=arguments.get("at"),
                    seconds_ago=arguments.get("seconds_ago"),
                    changes=arguments.get("changes"),
                )
                text = await _recorded(task, look, key=f"state:{round_}:{index}")
            elif name == CALL_ACTION_TOOL:
                # Not recorded: the call_key is what finds the child again on a resume.
                outcome = await call_action(
                    rekuest,
                    task,
                    str(arguments.get("action_id") or ""),
                    arguments.get("arguments") or {},
                    arguments.get("prompt"),
                    context,
                    call_key=f"call:{round_}:{index}",
                )
                attachments.extend(outcome.attachments)
                text = outcome.text
            elif name in (FIND_INSTALLABLE_TOOL, INSTALL_TOOL) and kabinet is None:
                text = "The app store is not reachable from here."
            elif name == FIND_INSTALLABLE_TOOL:
                search_store = partial(find_installable, kabinet, str(arguments.get("query") or ""))
                text = await _recorded(task, search_store, key=f"store:{round_}:{index}")
            elif name == INSTALL_TOOL:
                outcome = await install(
                    rekuest,
                    kabinet,
                    task,
                    str(arguments.get("definition_id") or ""),
                    arguments.get("prompt"),
                    call_key=f"install:{round_}:{index}",
                    report=report,
                )
                attachments.extend(outcome.attachments)
                text = outcome.text
            elif name in (REMEMBER_TOOL, RECALL_TOOL, FORGET_TOOL) and alpaka is None:
                text = "Memory is not available here."
            elif name == REMEMBER_TOOL:
                write = partial(
                    remember,
                    alpaka,
                    arguments.get("name"),
                    arguments.get("description"),
                    arguments.get("content"),
                    arguments.get("kind"),
                )
                text = await _recorded(task, write, key=f"remember:{round_}:{index}")
            elif name == RECALL_TOOL:
                read = partial(recall, alpaka, query=arguments.get("query"), name=arguments.get("name"))
                text = await _recorded(task, read, key=f"recall:{round_}:{index}")
            elif name == FORGET_TOOL:
                drop = partial(forget, alpaka, arguments.get("name"))
                text = await _recorded(task, drop, key=f"forget:{round_}:{index}")
            elif name in GRAPHQL_TOOLS and not engines:
                text = "No backend can be read from here."
            elif name in GRAPHQL_TOOLS:
                read_backend = partial(run_graphql_tool, name, arguments, engines)
                text = await _recorded(task, read_backend, key=f"graphql:{round_}:{index}")
            elif name == ASK_USER_TOOL:
                # Not recorded: aask records the answer itself.
                text = await ask_user(task, arguments.get("question"))
            else:
                text = f"Unknown tool '{name}'."
            messages.append(tool_result(call, name, clip(text, "Narrow the request.")))

    raise AssertionError("unreachable: the last round offers no tools")


# ---------------------------------------------------------------------------
# Wiring it to a room
# ---------------------------------------------------------------------------


def describe(message: Any) -> str:
    text = message.text or ""
    attached = getattr(message, "attached_structures", None)
    if attached:
        text += (
            f"\n[Attached: {', '.join(f'{s.identifier}:{s.object}' for s in attached)}]"
        )
    return text


def history_of(message: Message) -> list[dict]:
    """The prompt, then the room's conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # alpaka hands `before` newest first (so a page limit keeps the latest): read it oldest first.
    for previous in reversed(message.before or []):
        speaker = getattr(previous.agent, "name", None)
        messages.append(
            {
                "role": "assistant" if speaker == AGENT_ID else "user",
                "content": describe(previous),
            }
        )
    messages.append({"role": "user", "content": describe(message)})
    return messages


def context_of(message: Message) -> dict[str, str]:
    """What the room carries, identifier -> object; a newer message's attachment wins.

    "Segment this" usually points at an image attached a message or two ago.
    """
    context: dict[str, str] = {}
    # alpaka hands `before` newest first (so a page limit keeps the latest): read it oldest first.
    for carrier in [*reversed(message.before or []), message]:
        for structure in getattr(carrier, "attached_structures", None) or []:
            context[structure.identifier] = str(structure.object)
    return context

