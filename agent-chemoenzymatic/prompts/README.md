# `prompts/`

| Path | Contents |
|---|---|
| `core.md` | the system prompt (`core_lean.md` is the lean variant used for student models) |
| `sections/` | tool descriptions and policy sections loaded just in time per tool |
| `messages/` | nudge and status messages injected by the loop (buyability gate, repeated call, self-complete ignored, reaction ordering, …) |

Prompts are plain markdown with `{placeholders}` filled at send time. They are
part of the agent's behaviour: changing one changes the trajectories.
