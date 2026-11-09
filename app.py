"""kaller: a room assistant that runs actions. "Can you segment this?" finds a segmenter and calls it.

It searches the actions installed on the platform for one that does what was
asked, calls it, and answers in the room with what came back -- the structures the
action returned attached, so the room links straight to the mask, the table, the
new image.

**Only installed actions are offered.** The search asks the server for actions an
agent can take *now* (`assignable`), so the model never picks something nobody is
connected to run. Room replyers (anything taking an `@alpaka/message`, this app
included) are left out: handing "segment this" to another chatbot is not running it.

**"This" is what the room carries.** The structures attached to the message, and to
the messages before it (the newer one winning), fill the arguments that ask for
them by identifier. "Segment this" under an image fills the segmenter's
`@mikro/image` port without the model copying an id.

**Approval is the server's, not this app's.** Each call is a child of this turn.
Unless the turn runs in auto-mode, the server holds the child until the user
approves it in the room, showing the model's prompt next to what it
knows for certain (the action, its agent, the arguments). That prompt is an
`AssignPrompt`, written by the model: the `call_action` tool's `prompt` argument is
its title and description, field for field. A refusal comes back to
the model as a tool result, never as a failed turn. In auto-mode (the caller's
toggle: the root's `policy` is AUTO, and every child inherits it) the call runs at
once. This app never sets a policy.

**Returns are not expanded.** The app knows no structure but the message, so a
returned `@mikro/image` stays its id: the model reads `mask: @mikro/image 42`, and
the reply attaches it.

**What the data is can be read.** The model reads mikro, rekuest, alpaka and kabinet
through their GraphQL schemas, read-only, with this app's own clients: it searches a
schema, describes the types it needs and queries an attached structure by its id.
Mutations are refused; changing something is what actions are for.

**What an agent is doing can be read, now or at any time before.** `read_state` checks
out the state of every agent that runs an action -- where the stage is, which objective
is in -- so the model can look before it moves something, and look again after. Given a
time (`at`, or `seconds_ago`) it checks out the state as it was then, and lists the
changes recorded around it with their times and the tasks that made them: the model
walks the history by reading at those times.

**What is not installed can be.** The kabinet store knows what every release
provides; `find_installable` searches it, and `install` hands the definition to an
installer that is running (any action taking a `@kabinet/definition`, e.g. the
plugin engine's `auto_install`) as a held child, then waits until the action can
run. A release nobody approved yet is approved by the user: the installer finds
the app the user called this from among its callers (rekuest's `callerOf`) and
asks it -- orkestrator, when engaged, opens its install dialog. This app offers
nothing of the kind itself, so the installer looks past it.

**It remembers the person it works for.** Whoever assigned the turn -- the human at the
root of the task tree, the same when another assistant delegated it -- has memories in
alpaka: what they prefer, what they work on, how they want things done. The turn starts
with who they are and the index of what is remembered about them (one line each);
`recall` reads a memory whole, `remember` writes one (a name that exists is replaced),
`forget` drops one. alpaka hands them out only for the task's own provenance token, so
this app's login reaches nobody's memories but the assigner's of the turn it runs.

**It can ask.** `ask_user` holds the turn on a small blok in the room: a question and a
text field. The answer comes back as the tool's result; a dismissed question says so.

A turn is a workflow: every model round, search and wait is recorded, and every
call has its own `call_key`, so a turn taken over while a call waits for approval
finds that call again instead of asking twice.

Run:  arkitekt run dev
"""

from typing import Any

from alpaka import Alpaka, alpaka_service, stream_into_room
from alpaka.api.schema import LLMModel, Message
from arkitekt import App, Task, run
from kabinet import Kabinet, kabinet_service
from mikro import Mikro, mikro_service
from rekuest.arkitekt import rekuest_service
from rekuest.client.client import Rekuest

from kaller.assistant import (
    AGENT_ID,
    _recorded,
    attach_all,
    context_of,
    converse,
    history_of,
    memory_preamble,
)
from kaller.schema_engine import engines_for

__version__ = "0.1.0"

app = App(
    "kaller",
    __version__,
    description="A room assistant that finds and runs the actions a room asks for",
    author="jhnnsrs",
    services=[alpaka_service, rekuest_service, kabinet_service, mikro_service],
)


# It asks the questions the model writes and calls the actions it finds: both made at run
# time, so it declares both.
@app.workflow(dynamic_asks=True, dynamic_calls=True)
async def assign(
    message: Message,
    model: LLMModel,
    alpaka: Alpaka,
    rekuest: Rekuest,
    kabinet: Kabinet,
    mikro: Mikro,
    task: Task,
) -> Message:
    """Run the action a room asks for, installing it first if need be, and answer with what it returned.

    Each call and install waits for the user's approval in the room, unless the turn
    runs in auto-mode. Structures an action returns are attached to the answer.
    """

    def room_stream(room: str) -> Any:
        return stream_into_room(alpaka, room=room, agent_id=AGENT_ID)

    await task.aprogress(5, "Reading the room...")
    messages = history_of(message)

    async def recall_the_user() -> str:
        return await memory_preamble(alpaka)

    messages[0]["content"] += "\n\n" + await _recorded(task, recall_the_user, key="memory:index")
    turn = await converse(
        rekuest,
        alpaka.aopenai,
        room_stream,
        room=message.room.id,
        model=str(model.id),
        messages=messages,
        context=context_of(message),
        task=task,
        report=task.aprogress,
        kabinet=kabinet,
        alpaka=alpaka,
        engines=engines_for({"mikro": mikro, "rekuest": rekuest, "alpaka": alpaka, "kabinet": kabinet}),
    )
    await task.aprogress(100, "Done.")

    message_id = turn.message_id
    if message_id is None:
        # The answer was empty: say so, so the room is not left silent after a
        # turn that may have run something.
        async with room_stream(message.room.id) as reply:
            await reply.append(turn.answer)
            attach_all(reply, turn.attachments)
            message_id = reply.id

    return await alpaka.aget_message(id=message_id)


if __name__ == "__main__":
    run(app)
