# Doom Agent Practice

A university lab where you build an AI agent system that plays Doom autonomously.
The game backend (`tetsuo-doom`) is pre-built and exposes the game as an MCP server.
Your job is to extend the agent with memory, planning, additional agents, and better prompts.

## Architecture

```
main.py (agent loop)
  │
  ├── Commander agent   — reads player reports, issues objectives
  └── Player agent      — calls MCP tools to play the game
        │
        └── tetsuo-doom MCP server (game backend)
              └── ViZDoom engine (actual Doom game)
```

`main.py` runs two pydantic-ai agents in a loop. The player has access to the
Doom MCP tools (start game, explore, fight, etc.). After each turn the commander
reads the player's report and decides what to do next.

## Setup

Everything is pre-installed. Just sync the dependencies:

```bash
uv sync
```

Start the Doom MCP server (in a separate terminal, from inside `tetsuo-doom/`):

```bash
cd tetsuo-doom
uv run fastmcp run src/doom_mcp/server.py --transport sse
```

Run the agent:

```bash
uv run main.py
```

If you only use the local Ollama model and do not need the remote SSH tunnel,
leave `ENABLE_SSH_TUNNEL` unset. Set `ENABLE_SSH_TUNNEL=1` only if you really
need the remote host forwarding used by the original lab setup.

The agent also tries to start the local Doom MCP backend automatically from
`tetsuo-doom/` if it is not already running. Override the SSE URL with
`DOOM_MCP_URL` if needed, and set `AUTO_START_DOOM_MCP=0` to disable the
auto-start behavior.

The agent will open a Doom window, start exploring MAP01, and run for 10 commander
turns before stopping.

## What is already built

| File | What it does |
|------|-------------|
| `main.py` | Two-agent loop (commander + player), colored logging, prompt instructions |
| `tetsuo-doom/` | Full Doom MCP server exposing 19 tools (explore, fight, navigate, etc.) |

### Available Doom MCP tools

| Category | Tools |
|----------|-------|
| Game control | `start_game`, `new_episode`, `stop_game`, `get_available_actions` |
| State | `get_state`, `get_objects`, `get_threat_assessment`, `get_situation_report` |
| Navigation | `explore`, `move_to`, `get_navigation_info`, `get_map_knowledge`, `get_map` |
| Combat | `aim_and_shoot`, `strafe_and_shoot`, `retreat` |
| Autonomous | `set_objective`, `set_strategy` |

## Your Tasks

### 1. Memory MCP server (core task)

The agent runs out of context after a few turns because tool results (game state,
screenshots) are large. Build a **memory MCP server** that the player agent can
call to store and retrieve important information across turns.

Ideas for what to remember:
- Map layout: which rooms were visited, where enemies were found
- Enemy locations and types encountered
- Item locations (health packs, ammo, weapons)
- Which doors are open/closed and where keys were seen
- Events: kills, damage taken, objectives completed

Your memory server should expose tools like:
- `remember(key, value)` — store a fact
- `recall(key)` — retrieve a stored fact
- `recall_all()` — dump everything stored so far
- `summarize_progress()` — return a compact status for the commander

Start the memory server on a different port and add it to the player's toolset:

```python
doom_mcp = FastMCPToolset('http://localhost:8000/sse')
memory_mcp = FastMCPToolset('http://localhost:8001/sse')

doom_player = Agent(
    model,
    instructions=INSTRUCTIONS,
    toolsets=[doom_mcp, memory_mcp],
    capabilities=[hooks],
)
```

### 2. Planning tools

The commander currently gives one-line instructions. Add a planning MCP server
(or extend the memory server) with tools for longer-horizon planning:

- `set_goal(description)` — set the current high-level goal
- `get_goal()` — retrieve it
- `add_subtask(description)` — push a subtask onto a stack
- `complete_subtask()` — pop the current subtask and mark it done
- `get_plan()` — return the current goal + pending subtasks

This lets the commander build a multi-step plan and the player track progress
without repeating context in every prompt.

### 3. Add agents to the loop

The current loop is `commander → player → commander → ...`. You can insert
additional agents at any point to add capabilities.

**Example: add a tactician between commander and player**

The tactician receives the commander's high-level objective and breaks it down
into a precise sequence of tool calls before handing it to the player:

```python
tactician = Agent(
    model,
    instructions=TACTICIAN_INSTRUCTIONS,
    capabilities=[hooks],
)

async def main():
    result = await doom_player.run('Start playing!')
    while turn < 10:
        command = await commander.run(f'Player report: {result.output}')
        plan = await tactician.run(f'Objective: {command.output}')
        result = await doom_player.run(f'Execute this plan: {plan.output}')
        turn += 1
```

**Other agent roles you could add:**

- **Analyst** — reads raw tool output (game state JSON) and writes a concise
  natural-language summary before passing it to the commander, so the commander
  never sees large JSON blobs
- **Memory manager** — runs after each player turn, reads the tool results, and
  decides what to persist to the memory MCP server
- **Scout** — a second player agent with a restricted toolset (only `explore`,
  `get_navigation_info`, `get_map_knowledge`) that maps the level in advance
  while the main player fights
- **Critic** — reviews the commander's instruction before it reaches the player
  and rewrites it if it references a wrong tool name or an impossible action

Agents can share the same model or use different ones. Cheaper/faster models
work well for summarization; stronger models for strategy.

### 4. Prompt optimization

The current prompts in `INSTRUCTIONS` and `COMMANDER_INSTRUCTIONS` are a starting
point. Experiment with improving them:

- Make the player follow the gameplay loop more reliably
- Make the commander issue better-structured objectives
- Reduce token usage: trim verbose instructions that the model already follows
- Test whether shorter or longer instructions work better for your chosen model

Try different models by changing the `provider` variable in `main.py` (or
update `doom-bot.py` if you use that entrypoint):
```python
# Use Ollama (recommended for local inference): set `OLLAMA_BASE_URL` in `.env`
provider = "Ollama"   # uses gemma models via local Ollama (e.g. 'gemma4:latest')

# Or keep an OpenAI-compatible endpoint if you have one:
provider = "OpenAI"   # uses OPENAI_API_KEY and OPENAI_BASE_URL in `.env`
```

### 5. Ideas to explore (open-ended)

- **Episodic memory**: store a narrative summary of each completed level so the
  commander can learn from past runs
- **Map graph**: build a room-connectivity graph as the agent explores, and use
  it for pathfinding between known locations
- **Threat memory**: remember which enemy types appear in which rooms so the
  commander can pre-select the right weapon before entering
- **Skill library**: define high-level compound strategies (e.g. "clear room",
  "loot area", "find exit") that the commander can invoke by name

## How to add a new MCP server

1. Create `my_server.py` with a `FastMCP` app and your tools.
2. Run it in a separate terminal: `uv run fastmcp run my_server.py --transport sse`
3. Add it to the relevant agent's toolsets in `main.py`.

Minimal example:

```python
# my_server.py
from fastmcp import FastMCP

mcp = FastMCP("memory")

store = {}

@mcp.tool
def remember(key: str, value: str) -> str:
    store[key] = value
    return f"Stored: {key}"

@mcp.tool
def recall(key: str) -> str:
    return store.get(key, "Not found")
```

### Memory MCP server (example)

Start the simple in-repo memory MCP server (provides `remember`/`recall` and planning tools):

```bash
cd tetsuo-doom
uv run fastmcp run src/doom_mcp/memory_server.py --transport sse --port 8002
```

Then set the environment variable used by the agent runner (optional):

```bash
export MEMORY_MCP_URL=http://localhost:8002/sse    # Unix
setx MEMORY_MCP_URL "http://localhost:8002/sse"  # Windows (persist)
```

The `doom_player` in `doom-bot.py` is configured to use `MEMORY_MCP_URL`.


## Project structure

```
doom-gyak/
├── main.py              # Agent loop — edit this
├── pyproject.toml
├── .env                 # API keys (OPENAI_API_KEY) and endpoints (OLLAMA_BASE_URL)
└── tetsuo-doom/         # Doom MCP server — read-only reference
    └── src/doom_mcp/
        ├── server.py    # Tool definitions
        ├── game_manager.py
        ├── state.py
        ├── objects.py
        ├── navigation.py
        └── executor.py
```
