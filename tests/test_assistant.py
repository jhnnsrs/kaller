"""kaller's assistant: "segment this" finds an installed action, fills it from the room, and runs it.

Nothing reaches a server or a model: the model is a scripted stream, rekuest finds one
fake segmenter (and one room replyer it must leave out), and the task's ``acall_raw``
stands in for the server's hold (it returns, or raises ``DeniedError`` as a refused
child does).
"""

import asyncio
import datetime
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from arkitekt import AssignPrompt, DeniedError, PortKind
from rath.operation import GraphQLException

from kaller import assistant as A


def port(
    key: str, kind: PortKind, identifier: str | None = None, nullable: bool = False, default: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(
        key=key, kind=kind, identifier=identifier, nullable=nullable, default=default, description=None, children=()
    )


SEGMENT = SimpleNamespace(
    id="7",
    name="Segment nuclei",
    description="Finds the nuclei in an image.",
    args=(port("image", PortKind.STRUCTURE, "@mikro/image"), port("sigma", PortKind.FLOAT, default=2.0)),
    returns=(port("mask", PortKind.STRUCTURE, "@mikro/image"), port("count", PortKind.INT)),
)
CHATBOT = SimpleNamespace(
    id="8",
    name="Room assistant",
    description="Answers questions about nuclei.",
    args=(port("message", PortKind.STRUCTURE, A.MESSAGE),),
    returns=(),
)


class FakeRekuest:
    def __init__(self) -> None:
        self.filters: list[Any] = []

    async def afind(self, id: str) -> Any:
        return {"7": SEGMENT, "8": CHATBOT}[id]

    async def alist_actions(self, filters: Any, pagination: Any) -> list[Any]:
        self.filters.append(filters)
        return [SEGMENT, CHATBOT] if "nuclei" in (filters.search or "") else []


class FakeTask:
    """Answers a child call as the server would after the user decided (or in auto-mode)."""

    def __init__(self, deny: bool = False, denied: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.deny = deny
        self.denied = denied

    async def acall_raw(
        self, *, action: Any, kwargs: dict, prompt: AssignPrompt | None = None, call_key: str | None = None
    ) -> Any:
        self.calls.append({"action": action.id, "kwargs": kwargs, "prompt": prompt, "call_key": call_key})
        if self.deny:
            raise DeniedError(self.denied, task="99")
        return {"mask": "42", "count": 12}

    async def arecord(self, fn: Any, key: str | None = None) -> Any:
        value = fn()
        return await value if asyncio.iscoroutine(value) else value


def structure(identifier: str, object_: int) -> SimpleNamespace:
    return SimpleNamespace(identifier=identifier, object=object_)


def room_message(text: str, attached: list | None = None, before: list | None = None, agent: str = "user") -> Any:
    return SimpleNamespace(
        text=text,
        attached_structures=attached or [],
        before=before or [],
        agent=SimpleNamespace(name=agent),
        room=SimpleNamespace(id="r1"),
    )


# -- finding ------------------------------------------------------------------ #


def test_find_actions_asks_for_installed_ones_and_leaves_out_room_replyers() -> None:
    rekuest = FakeRekuest()
    text = asyncio.run(A.find_actions(rekuest, "segment nuclei"))

    assert rekuest.filters[0].assignable is True
    assert "Segment nuclei (id: 7)" in text and "image: STRUCTURE @mikro/image" in text
    assert "mask: STRUCTURE @mikro/image" in text
    assert "Room assistant" not in text
    assert "No installed action" in asyncio.run(A.find_actions(rekuest, "teleport"))


# -- calling ------------------------------------------------------------------ #


def test_an_earlier_attachment_fills_the_image_and_a_newer_one_wins() -> None:
    older = room_message("here is plate 1", attached=[structure("@mikro/image", 30)])
    newer = room_message("and plate 2", attached=[structure("@mikro/image", 31)])
    # As alpaka hands `before`: newest first.
    message = room_message("can you segment this?", before=[newer, older])

    assert A.context_of(message) == {"@mikro/image": "31"}


def test_the_model_reads_the_room_oldest_first() -> None:
    first = room_message("segment plate 1")
    answer = room_message("Done: 12 nuclei.", agent=A.AGENT_ID)
    second = room_message("now plate 2")
    # As alpaka hands `before`: newest first.
    message = room_message("and plate 3", before=[second, answer, first])

    history = A.history_of(message)

    assert [(m["role"], m["content"]) for m in history[1:]] == [
        ("user", "segment plate 1"),
        ("assistant", "Done: 12 nuclei."),
        ("user", "now plate 2"),
        ("user", "and plate 3"),
    ]


def test_an_approved_call_runs_with_the_models_prompt_and_attaches_what_it_returned() -> None:
    task = FakeTask()
    outcome = asyncio.run(
        A.call_action(
            FakeRekuest(),
            task,
            "7",
            {},
            {"title": "Segment nuclei in plate 2", "description": "You asked for the nuclei in the image you attached."},
            {"@mikro/image": "31"},
            "call:0:0",
        )
    )

    [call] = task.calls
    assert call["kwargs"] == {"image": {"__identifier": "@mikro/image", "object": "31"}}
    assert call["call_key"] == "call:0:0"
    assert call["prompt"] == AssignPrompt(
        title="Segment nuclei in plate 2", description="You asked for the nuclei in the image you attached."
    )
    assert outcome.attachments == [("@mikro/image", 42)]
    assert "mask: @mikro/image 42" in outcome.text and "count: 12" in outcome.text


def test_a_prompt_without_a_title_or_flattened_to_a_string_still_reaches_the_server() -> None:
    assert A.prompt_of({"description": " why "}, "Run Segment nuclei") == AssignPrompt(
        title="Run Segment nuclei", description="why"
    )
    assert A.prompt_of("because you asked", "Run Segment nuclei") == AssignPrompt(
        title="Run Segment nuclei", description="because you asked"
    )
    assert A.prompt_of(None, "Run Segment nuclei") == AssignPrompt(title="Run Segment nuclei")


def test_a_denied_call_is_a_tool_result_saying_nothing_ran() -> None:
    outcome = asyncio.run(
        A.call_action(
            FakeRekuest(), FakeTask(deny=True, denied="not that one"), "7", {}, "why", {"@mikro/image": "31"}, "k"
        )
    )
    assert outcome.text == 'The user declined to let Segment nuclei run; nothing ran. They said: "not that one"'
    assert outcome.attachments == []


def test_nothing_is_called_without_the_image_or_with_a_made_up_argument_or_on_a_replyer() -> None:
    task = FakeTask()
    missing = asyncio.run(A.call_action(FakeRekuest(), task, "7", {}, "why", {}, "k"))
    unknown = asyncio.run(A.call_action(FakeRekuest(), task, "7", {"radius": 3}, "why", {"@mikro/image": "1"}, "k"))
    replyer = asyncio.run(A.call_action(FakeRekuest(), task, "8", {}, "why", {A.MESSAGE: "1"}, "k"))

    assert "needs: image" in missing.text
    assert "no argument radius" in unknown.text
    assert "answers room messages" in replyer.text
    assert task.calls == []


# -- reading state ------------------------------------------------------------ #


NOW = datetime.datetime(2026, 10, 20, 12, 0, tzinfo=datetime.UTC)


def change(minutes_ago: int, path: str, task: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        op="replace", interface="stage", path=path, timestamp=NOW - datetime.timedelta(minutes=minutes_ago), task=task
    )


MOVE = SimpleNamespace(id="12", action=SimpleNamespace(name="Move stage"))


class StateRekuest(FakeRekuest):
    """Two agents run the segmenter (one twice over); the second one's state cannot be read.

    The first one's stage moved 30 and 10 minutes ago; a checkout before that sees it at 0.
    """

    def __init__(self) -> None:
        super().__init__()
        self.implementation_filters: list[Any] = []
        self.checkouts: list[dict[str, Any]] = []

    async def alist_implementations(self, filters: Any, pagination: Any) -> list[Any]:
        self.implementation_filters.append(filters)
        if list(filters.action.ids) != ["7"]:
            return []
        return [SimpleNamespace(agent=SimpleNamespace(id=agent)) for agent in ("a1", "a1", "a2")]

    async def acheckout_agent(
        self,
        agent: str,
        timestamp: datetime.datetime | None = None,
        forward_patch_count: int = 0,
        backward_patch_count: int = 0,
    ) -> SimpleNamespace:
        """``RekuestApi.acheckout_agent``'s arguments, recorded."""
        self.checkouts.append(
            {"agent": agent, "timestamp": timestamp, "backward_patch_count": backward_patch_count, "forward_patch_count": forward_patch_count}
        )
        if agent == "a2":
            raise GraphQLException("No snapshot found for state 3")
        moves = [change(30, "/x", MOVE), change(10, "/y")]
        when = timestamp or NOW
        before = [m for m in moves if m.timestamp <= when]
        after = [m for m in moves if m.timestamp > when] if timestamp else []
        x = 1.5 if len(before) >= 1 else 0.0
        y = 2.0 if len(before) >= 2 else 0.0
        return SimpleNamespace(
            agent_id=agent, values={"stage": {"x": x, "y": y}, "objective": "60x"},
            backward_patches=before, forward_patches=after,
        )


def test_read_state_checks_out_each_agent_running_the_action_once() -> None:
    rekuest = StateRekuest()
    text = asyncio.run(A.read_state(rekuest, "7", now=NOW))

    [filters] = rekuest.implementation_filters
    assert filters.active is True
    assert [c["agent"] for c in rekuest.checkouts] == ["a1", "a2"]
    assert rekuest.checkouts[0] == {
        "agent": "a1", "timestamp": None, "backward_patch_count": A.STATE_CHANGES, "forward_patch_count": 0
    }
    assert text.splitlines() == [
        "It is 2026-10-20T12:00:00+00:00 now. Current state:",
        "- agent a1:",
        '    stage: {"x": 1.5, "y": 2.0}',
        '    objective: "60x"',
        "  last changes, oldest first:",
        "    2026-10-20T11:30:00+00:00  replace stage/x  (task 12: Move stage)",
        "    2026-10-20T11:50:00+00:00  replace stage/y",
        "- agent a2: its state could not be read: No snapshot found for state 3",
    ]
    assert "No connected agent" in asyncio.run(A.read_state(rekuest, "8", now=NOW))


def test_read_state_travels_back_to_any_earlier_time() -> None:
    rekuest = StateRekuest()
    by_seconds = asyncio.run(A.read_state(rekuest, "7", seconds_ago=20 * 60, changes=3, now=NOW))
    by_time = asyncio.run(A.read_state(rekuest, "7", at="2026-10-20T11:40:00", now=NOW))

    assert rekuest.checkouts[0] == {
        "agent": "a1",
        "timestamp": NOW - datetime.timedelta(minutes=20),
        "backward_patch_count": 3,
        "forward_patch_count": 3,
    }
    assert by_seconds == by_time  # a time without a zone is UTC
    assert "State at 2026-10-20T11:40:00+00:00:" in by_time
    assert '    stage: {"x": 1.5, "y": 0.0}' in by_time.splitlines()
    up_to, after = by_time.split("  changes after, oldest first:\n")
    assert "stage/x  (task 12: Move stage)" in up_to and "stage/y" in after


def test_read_state_refuses_a_time_it_cannot_use() -> None:
    rekuest = StateRekuest()
    assert "in the future" in asyncio.run(A.read_state(rekuest, "7", at="2027-01-01T00:00:00Z", now=NOW))
    assert "not an ISO time" in asyncio.run(A.read_state(rekuest, "7", at="yesterday", now=NOW))
    assert "not a number" in asyncio.run(A.read_state(rekuest, "7", seconds_ago="a while", now=NOW))
    assert rekuest.checkouts == []


def test_a_turn_reads_the_state_before_answering() -> None:
    model = ScriptedModel(
        [
            [chunk(call={"id": "c1", "name": A.READ_STATE_TOOL, "arguments": {"action_id": "7"}})],
            [chunk("The stage is at 1.5, 2.0.")],
        ]
    )
    room = Room()
    message = room_message("where is the stage?")

    turn = asyncio.run(
        A.converse(
            StateRekuest(),
            model,
            room.stream,
            room="r1",
            model="m",
            messages=A.history_of(message),
            context={},
            task=FakeTask(),
        )
    )

    [read] = [m for m in model.sent[1] if m.get("role") == "tool"]
    assert read["name"] == A.READ_STATE_TOOL and '"x": 1.5' in read["content"]
    assert turn.answer == "The stage is at 1.5, 2.0."


# -- the whole turn: the model finds, calls, then answers --------------------- #


def chunk(content: str | None = None, call: dict | None = None) -> SimpleNamespace:
    tool_calls = None
    if call is not None:
        tool_calls = [
            SimpleNamespace(
                index=0,
                id=call["id"],
                function=SimpleNamespace(name=call["name"], arguments=json.dumps(call["arguments"])),
            )
        ]
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


class ScriptedModel:
    """Answers each completion with the next scripted round, and keeps what it was sent."""

    def __init__(self, rounds: list[list[SimpleNamespace]]) -> None:
        self.rounds = rounds
        self.sent: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, model: str, messages: list[dict], tools: Any, stream: bool) -> Any:
        self.sent.append(list(messages))
        chunks = self.rounds.pop(0)

        async def stream_() -> Any:
            for c in chunks:
                yield c

        return stream_()


class Room:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list]] = []

    def stream(self, room: str) -> Any:
        @asynccontextmanager
        async def opened() -> Any:
            parts: list[str] = []
            attached: list = []

            async def append(token: str) -> None:
                parts.append(token)

            yield SimpleNamespace(id="m1", append=append, attach=attached.append)
            self.messages.append(("".join(parts), attached))

        return opened()


@pytest.mark.parametrize("deny", [False, True])
def test_a_turn_finds_calls_and_answers_with_the_mask_attached(deny: bool) -> None:
    model = ScriptedModel(
        [
            [chunk(call={"id": "c1", "name": A.FIND_ACTIONS_TOOL, "arguments": {"query": "segment nuclei"}})],
            [
                chunk(
                    call={
                        "id": "c2",
                        "name": A.CALL_ACTION_TOOL,
                        "arguments": {"action_id": "7", "prompt": {"title": "Segment nuclei in plate 2"}},
                    }
                )
            ],
            [chunk("Okay, I left it." if deny else "Found 12 nuclei.")],
        ]
    )
    room = Room()
    task = FakeTask(deny=deny)
    message = room_message("can you segment this?", attached=[structure("@mikro/image", 31)])

    turn = asyncio.run(
        A.converse(
            FakeRekuest(),
            model,
            room.stream,
            room="r1",
            model="m",
            messages=A.history_of(message),
            context=A.context_of(message),
            task=task,
        )
    )

    [found] = [m for m in model.sent[1] if m.get("role") == "tool"]
    assert "Segment nuclei (id: 7)" in found["content"]
    called = [m for m in model.sent[2] if m.get("role") == "tool"][-1]
    assert ("declined" in called["content"]) is deny
    assert [c["call_key"] for c in task.calls] == ["call:1:0"]

    [(text, attached)] = room.messages
    assert text == turn.answer
    assert [(s.identifier, s.object) for s in attached] == ([] if deny else [("@mikro/image", 42)])


# -- installing what is not --------------------------------------------------- #


INSTALLABLE = SimpleNamespace(
    id="d1",
    hash="segment-hash",
    name="Segment nuclei",
    description="Finds the nuclei in an image.",
    flavours=(
        SimpleNamespace(id="f1", release=SimpleNamespace(id="5", version="1.2.0", app=SimpleNamespace(identifier="segmenter"))),
    ),
)
AUTO_INSTALL = SimpleNamespace(
    id="20",
    name="Auto Install",
    description="Installs what provides a definition.",
    args=(port("definition", PortKind.STRUCTURE, A.DEFINITION),),
    returns=(port("return0", PortKind.STRUCTURE, "@kabinet/pod"),),
)


class FakeKabinet:
    def __init__(self) -> None:
        self.searched: list[str] = []

    async def asearch_installable(self, search: str, limit: int | None = None) -> tuple:
        self.searched.append(search)
        return (INSTALLABLE,) if "nuclei" in search else ()

    async def aget_installable(self, id: str) -> Any:
        assert id == INSTALLABLE.id
        return INSTALLABLE


class StoreRekuest:
    """A rekuest where the installer is running, and the segmenter comes online after ``online_after`` looks."""

    def __init__(self, installer: Any = AUTO_INSTALL, online_after: int = 1) -> None:
        self.installer = installer
        self.looks = 0
        self.online_after = online_after
        self.demands: list[Any] = []

    async def afind(self, hash: str) -> Any:
        assert hash == INSTALLABLE.hash
        return SEGMENT

    async def alist_actions(self, filters: Any, pagination: Any) -> list[Any]:
        if filters.demands:
            self.demands.append(filters)
            return [self.installer] if self.installer else []
        assert list(filters.ids) == [SEGMENT.id] and filters.assignable is True
        self.looks += 1
        return [SEGMENT] if self.looks > self.online_after else []


class InstallTask(FakeTask):
    async def acall_raw(self, *, action: Any, kwargs: dict, prompt: AssignPrompt | None = None, call_key: str | None = None) -> Any:
        self.calls.append({"action": action.id, "kwargs": kwargs, "prompt": prompt, "call_key": call_key})
        if self.deny:
            raise DeniedError(self.denied, task="99")
        return {"return0": "77"}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "INSTALL_POLL_SECONDS", 0)


def test_find_installable_lists_definitions_with_the_releases_providing_them() -> None:
    kabinet = FakeKabinet()
    text = asyncio.run(A.find_installable(kabinet, "segment nuclei"))

    assert "Segment nuclei (definition id: d1)" in text
    assert "segmenter 1.2.0 (release 5)" in text
    assert "Nothing in the app store" in asyncio.run(A.find_installable(kabinet, "teleport"))


def test_install_hands_the_definition_to_a_running_installer_and_waits_until_it_can_run() -> None:
    rekuest, task = StoreRekuest(online_after=2), InstallTask()

    outcome = asyncio.run(
        A.install(rekuest, FakeKabinet(), task, "d1", {"title": "Install the nuclei segmenter"}, "install:0:0")
    )

    [demand] = rekuest.demands
    takes, returns = demand.demands
    assert [(m.kind, m.identifier) for m in takes.matches] == [(PortKind.STRUCTURE, A.DEFINITION)]
    assert [(m.kind, m.identifier) for m in returns.matches] == [(PortKind.STRUCTURE, A.POD)]
    assert demand.assignable is True
    [call] = task.calls
    assert call == {
        "action": "20",
        "kwargs": {"definition": "d1"},
        "prompt": AssignPrompt(title="Install the nuclei segmenter"),
        "call_key": "install:0:0",
    }
    assert outcome.text.startswith("Segment nuclei is installed and running:") and "(id: 7)" in outcome.text
    assert outcome.attachments == [("@kabinet/pod", 77)]


def test_what_is_already_running_is_not_installed_again() -> None:
    task = InstallTask()
    outcome = asyncio.run(A.install(StoreRekuest(online_after=0), FakeKabinet(), task, "d1", None, "k"))

    assert "already installed" in outcome.text and task.calls == []


def test_without_an_installer_the_release_is_attached_for_the_user() -> None:
    outcome = asyncio.run(A.install(StoreRekuest(installer=None), FakeKabinet(), InstallTask(), "d1", None, "k"))

    assert "No installer is running" in outcome.text
    assert outcome.attachments == [("@kabinet/release", 5)]


def test_a_declined_install_installs_nothing() -> None:
    task = InstallTask(deny=True, denied="not now")
    outcome = asyncio.run(A.install(StoreRekuest(), FakeKabinet(), task, "d1", None, "k"))

    assert outcome.text == 'The user declined to install Segment nuclei; nothing was installed. They said: "not now"'


def test_an_install_that_never_comes_online_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "INSTALL_TIMEOUT_SECONDS", 0)
    outcome = asyncio.run(A.install(StoreRekuest(online_after=10**6), FakeKabinet(), InstallTask(), "d1", None, "k"))

    assert "has not come online" in outcome.text
    assert outcome.attachments == [("@kabinet/pod", 77)]


# -- remembering the user, and asking them ------------------------------------ #


class FakeMemories:
    """alpaka's memory surface for one assigner: the store the turn reads and writes."""

    def __init__(self, refuse: str | None = None) -> None:
        self.store: dict[str, SimpleNamespace] = {}
        self.refuse = refuse

    def _check(self) -> None:
        if self.refuse:
            raise GraphQLException(self.refuse)

    async def aassigner(self) -> SimpleNamespace:
        self._check()
        return SimpleNamespace(sub="ada", preferred_username="ada", name="Ada Lovelace")

    async def amemory_index(self, limit: int) -> tuple:
        self._check()
        return tuple(self.store.values())[:limit]

    async def aremember(self, name: str, description: str, content: str, kind: Any) -> SimpleNamespace:
        self._check()
        self.store[name] = SimpleNamespace(name=name, description=description, content=content, kind=kind)
        return self.store[name]

    async def arecall(self, search: str | None = None, names: list[str] | None = None, limit: int = 5) -> tuple:
        self._check()
        if names:
            return tuple(m for n, m in self.store.items() if n in names)[:limit]
        return tuple(m for m in self.store.values() if (search or "").lower() in (m.description + m.content).lower())[:limit]

    async def aforget(self, name: str) -> bool:
        self._check()
        return self.store.pop(name, None) is not None


class AskingTask(FakeTask):
    """Answers ``adynamic_ask`` as the user would, and keeps what it was asked."""

    def __init__(self, answer: str | None) -> None:
        super().__init__()
        self.answer = answer
        self.asked: list[tuple[Any, str, dict]] = []

    async def adynamic_ask(self, blok: Any, *, args: dict | None = None, returns: Any = None) -> dict[str, Any]:
        self.asked.append((blok, blok.name, args or {}))
        if self.answer is None:
            raise A.DismissedError("dismissed")
        return {"answer": self.answer}


def test_the_turn_starts_with_who_assigned_it_and_what_is_remembered() -> None:
    memories = FakeMemories()
    asyncio.run(memories.aremember("export-format", "Exports as OME-TIFF", "Never PNG: they lose the metadata.", A.MemoryKind.FEEDBACK))

    preamble = asyncio.run(A.memory_preamble(memories))

    assert "You work for Ada Lovelace." in preamble
    assert "- export-format (FEEDBACK): Exports as OME-TIFF" in preamble
    assert "Never PNG" not in preamble, "the index lists, recall reads"


def test_without_a_task_token_the_turn_goes_on_without_memory() -> None:
    preamble = asyncio.run(A.memory_preamble(FakeMemories(refuse="Memories are reached only from a task")))
    assert preamble.startswith("Memory is not available in this turn")


def test_a_turn_remembers_recalls_and_forgets_under_names() -> None:
    memories = FakeMemories()
    model = ScriptedModel(
        [
            [chunk(call={"id": "c1", "name": A.REMEMBER_TOOL, "arguments": {"name": "export-format", "description": "Exports as OME-TIFF", "content": "Never PNG.", "kind": "feedback"}})],
            [chunk(call={"id": "c2", "name": A.RECALL_TOOL, "arguments": {"name": "export-format"}})],
            [chunk(call={"id": "c3", "name": A.FORGET_TOOL, "arguments": {"name": "export-format"}})],
            [chunk("Done.")],
        ]
    )

    asyncio.run(
        A.converse(
            FakeRekuest(), model, Room().stream, room="r1", model="m",
            messages=A.history_of(room_message("I always want OME-TIFF")), context={}, task=FakeTask(), alpaka=memories,
        )
    )

    results = [m["content"] for m in model.sent[-1] if m.get("role") == "tool"]
    assert results[0] == "Remembered 'export-format'."
    assert "## export-format (FEEDBACK)" in results[1] and "Never PNG." in results[1]
    assert results[2] == "Forgot 'export-format'." and memories.store == {}


def test_a_memory_of_an_unknown_kind_or_without_content_is_not_written() -> None:
    memories = FakeMemories()
    assert "kind is one of" in asyncio.run(A.remember(memories, "x", "y", "z", "GOSSIP"))
    assert "needs a name" in asyncio.run(A.remember(memories, "x", "", "z", "USER"))
    assert memories.store == {}


@pytest.mark.parametrize("answer", ["Plate 2, please", None])
def test_asking_holds_the_turn_on_a_question_blok(answer: str | None) -> None:
    task = AskingTask(answer)

    text = asyncio.run(A.ask_user(task, "Which plate?"))

    [(blok, message, args)] = task.asked
    assert (message, args) == ("Which plate?", {"question": "Which plate?"})
    assert blok.source == A.ASK_BLOK and [local.key for local in blok.locals] == ["answer"]
    assert [ret.key for ret in blok.returns] == ["answer"]
    if answer:
        assert text == "The user answered: Plate 2, please"
    else:
        assert text.startswith("The user dismissed the question")


# -- reading a backend -------------------------------------------------------- #


class FakeEngine:
    name = "mikro"
    description = "microscopy data"

    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []

    async def execute(self, query: str, variables: Any = None) -> str:
        self.queries.append((query, variables))
        return '{"image":{"name":"plate 2"}}'


def test_a_graphql_query_reaches_the_engine_and_its_data_the_model() -> None:
    engine = FakeEngine()
    query = 'query { image(id: "31") { name } }'
    model = ScriptedModel(
        [
            [chunk(call={"id": "c1", "name": "graphql_query", "arguments": {"service": "mikro", "query": query}})],
            [chunk("That is plate 2.")],
        ]
    )
    room = Room()
    message = room_message("what is this?", attached=[structure("@mikro/image", 31)])

    turn = asyncio.run(
        A.converse(
            FakeRekuest(),
            model,
            room.stream,
            room="r1",
            model="m",
            messages=A.history_of(message),
            context=A.context_of(message),
            task=FakeTask(),
            engines={"mikro": engine},
        )
    )

    assert engine.queries == [(query, None)]
    [read] = [m for m in model.sent[1] if m.get("role") == "tool"]
    assert read["name"] == "graphql_query" and "plate 2" in read["content"]
    assert turn.answer == "That is plate 2."
