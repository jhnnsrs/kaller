# kaller

A room assistant that runs actions. "Can you segment this?" finds a segmenter installed on
the platform, fills its image from what the room has attached, and calls it. The answer goes
back to the room with whatever the action returned attached.

- **Only installed actions.** It searches for actions a connected agent can run now. Other
  room assistants are left out.
- **The user approves every call.** Each call is a held child of the turn. The user sees the
  prompt the model wrote, the action, its agent and the arguments. In auto-mode, which the
  caller turns on, the call runs right away.
- **It reads the data.** mikro, rekuest, alpaka and kabinet are searchable through their
  GraphQL schemas, read-only, with the app's own clients.
- **It reads agent state** (`read_state`), now or at any earlier time.
- **It installs what is missing.** It searches the app store and hands the definition to a
  running installer.
- **It remembers the person it works for** (alpaka memories) and **can ask them** a question
  in the room.

Run it against a server:

```sh
arkitekt run dev
```

Tests (offline: the model, the room and every client are fakes):

```sh
uv run pytest
```
