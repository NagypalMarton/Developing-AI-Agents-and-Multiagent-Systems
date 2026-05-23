import asyncio
import json
import logging
import os
from dotenv import load_dotenv
load_dotenv()

from sshtunnel import SSHTunnelForwarder

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, show_path=False, markup=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("doom")

from dataclasses import replace

from pydantic_ai import Agent, ModelRequestContext, ModelResponse, RunContext, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.messages import (
    TextPart, ThinkingPart, ToolCallPart,
    ToolReturnPart, RetryPromptPart, UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(_ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    for msg in request_context.messages:
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                pass  # log.info("[cyan]USER[/cyan]  %s", part.content)
            elif isinstance(part, ToolReturnPart):
                log.info("[green]TOOL RETURN[/green]  [bold]%s[/bold] → %s", part.tool_name, escape(str(part.content)[:500]))
            elif isinstance(part, RetryPromptPart):
                log.warning("[red]RETRY[/red]  [bold]%s[/bold]: %s", part.tool_name, escape(str(part.content)))
    return request_context

@hooks.on.after_model_request
async def log_response(_ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse:
    for part in response.parts:
        if isinstance(part, ThinkingPart):
            log.info("[dim italic]THINKING[/dim italic]\n[dim]%s[/dim]", escape(part.content))
        elif isinstance(part, ToolCallPart):
            log.info("[yellow]TOOL CALL[/yellow]  [bold]%s[/bold]  args=%s", part.tool_name, escape(part.args))
        #elif isinstance(part, TextPart):
            #log.info("[white]TEXT[/white]  %s", escape(part.content))
    return response

@hooks.on.before_tool_execute
async def log_tool_call(_ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs:
    log.info("[yellow]EXECUTE[/yellow]  [bold]%s[/bold]  args=%s", call.tool_name, escape(str(args)))
    return args

@hooks.on.after_tool_execute
async def log_tool_result(_ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, result: object) -> object:
    log.info("[green]RESULT[/green]  [bold]%s[/bold] → %s", call.tool_name, escape(str(result)[:500]))
    return result



tunnel = SSHTunnelForwarder(
    ("spark.mit.bme.hu", 10222),
    ssh_username="uname",
    ssh_pkey="/Users/uname/.ssh/id_rsa",
    ssh_private_key_password=os.getenv("SSH_KEY_PASSWORD"),
    remote_bind_address=("localhost", 8000),
    local_bind_address=("localhost", 8000),
)

tunnel.start()
model = OpenAIChatModel(
    model_name='google/gemma-4-31B-it',  # pl. 'qwen2.5-7b-instruct'
    provider=OpenAIProvider(
        base_url=os.getenv('OPENAI_BASE_URL', "http://spark.mit.bme.hu:8888/v1"),
        api_key=os.getenv('OPENAI_API_KEY'),
    ),
)



'''
provider = "OpenAI"
if provider == "OpenAI":
    model = OpenAIChatModel('gpt-5.4-mini')
else:
    _provider = OllamaProvider(base_url='http://localhost:11434/v1')
    _profile = replace(_provider.model_profile('gemma4:latest'), openai_chat_send_back_thinking_parts='tags')
    model = OllamaModel('gemma4:latest', provider=_provider, profile=_profile)
'''

INSTRUCTIONS = """
You are an autonomous Doom player controlled turn-by-turn by a commander.
Each turn you receive ONE objective. Execute it with the minimum tool calls needed, then stop and report.
You MUST use your tools — do not describe actions, execute them.

## Startup
If no game is running, call start_game(wad="freedoom2", map_name="MAP02", window_visible=true, difficulty=3).
Call start_game ALONE — do not call any other tool in the same turn.
Wait for start_game to return before calling explore or any other tool.
Do not call start_game again unless the game crashes.

## One task per turn
Execute exactly the objective given by the commander. Do not chain multiple objectives.
- If told to explore: call explore() once, then report what you found.
- If told to fight an enemy: call get_threat_assessment(), then aim_and_shoot or strafe_and_shoot once, then report.
- If told to collect an item: call move_to(object_id=<id>) once, then report.
- If told to open a door or use a switch: call move_to(object_id=<id>, use=true) once, then report.
Stop after completing the task. Do not keep exploring or fighting after the objective is done.

## Exploration
- explore(stop_on_enemy=true) — walks until an enemy appears. Report what you see.
- explore(stop_on_item=true) — walks until a health/ammo/weapon appears. Report what you see.
- If explore returns stop_reason="stuck" or "max_tics": call get_navigation_info() to find a new direction, then report it.

## Combat
- get_threat_assessment() — returns enemy IDs and priorities.
- strafe_and_shoot(object_id=<id>) — use against hitscan enemies (chaingunner, former human).
- aim_and_shoot(object_id=<id>) — use against all other enemies.
- retreat() — move away when health is critical.

## Doors and keys
Doom doors are opened by walking up and using them. Keys unlock color-coded locked doors.
- Doors: call move_to(object_id=<door_id>, use=true). If the door does not open, it requires a key.
- Keys (RedCard, BlueCard, YellowCard, RedSkull, BlueSkull, YellowSkull): visible in get_objects() with type="key".
  Collect a key with move_to(object_id=<key_id>). After collecting, the matching locked door can be opened.
- Locked doors are named things like "Door, Red Key" or "Door, Blue Key" — match key color to door color.
- If you spot a key or locked door, report its object_id and color so the commander can plan.

## Items and switches
- Collect items (health, ammo, weapons): move_to(object_id=<id>).
- Activate switches: move_to(object_id=<id>, use=true).

## Exit
- Scan get_objects() for names containing "exit", "switch", or "teleport".
- Call move_to(object_id=<id>, use=true) to finish the level.
- When episode_finished=true in any result, call new_episode() to advance to the next map.

## Rules
- ALWAYS call a tool. Never describe what you would do.
- Never call stop_game.
- Never call take_action directly.
- One objective per turn — stop after completing it.

## After every turn
Write a report in exactly this format:

  REPORT: <what you did — one action, its outcome>
  STATE: health=<hp> armor=<armor> ammo=<n> kills=<k>
  SITUATION: <what is visible now — enemies, items, doors, keys nearby with object IDs>
  NEXT: <what you think the commander should order next>
"""

COMMANDER_INSTRUCTIONS = """
You are a Doom commander directing an AI player agent that controls the game
via MCP tools. Your job is to read the player's status report, evaluate progress,
and issue the next clear, prioritized objective.

## Available player tools (exact names)
Exploration:   explore, get_navigation_info, get_map_knowledge, get_map
State:         get_state, get_objects, get_threat_assessment, get_situation_report
Combat:        aim_and_shoot, strafe_and_shoot, retreat
Movement:      move_to
Game control:  start_game, new_episode, get_available_actions

## Your role
- You do NOT play the game. You do NOT call any tools yourself.
- The player reports what happened after each batch of actions.
- You respond with one focused instruction telling the player what to do next.

## How to evaluate the player's report
- Health below 40: prioritize retreat + explore(stop_on_item=true) to find health.
- Active enemies spotted: prioritize aim_and_shoot or strafe_and_shoot.
- No enemies, healthy: direct the player to explore(stop_on_enemy=true).
- Exit or switch found: direct move_to(object_id, use=true) to activate it.
- Stuck / max_tics: direct get_navigation_info then explore in a different direction.
- episode_finished seen: direct new_episode() immediately.

## How to give instructions
Respond in exactly this format:

  OBJECTIVE: <what to do>
  REASON: <why, based on the player's report>
  HINT: <exact tool call to use, e.g. explore(stop_on_enemy=true) or aim_and_shoot(object_id=42)>

Examples:
  OBJECTIVE: Hunt for enemies.
  REASON: Area is clear and exploration progress is low.
  HINT: Call explore(stop_on_enemy=true) — when stop_reason="enemy_spotted", use aim_and_shoot(object_id=<id from get_threat_assessment>).

  OBJECTIVE: Activate the exit switch.
  REASON: No enemies remain and you can see an exit switch in get_objects.
  HINT: Call move_to(object_id=<switch_id>, use=true).

  OBJECTIVE: Find health — HP is critical.
  REASON: Health is below 30.
  HINT: Call retreat() then explore(stop_on_item=true); when stop_reason="item_found" call move_to(object_id=<id>).

Keep instructions short and actionable. One objective per turn.
"""

doom_mcp = FastMCPToolset('http://localhost:8001/sse', max_retries=3)
doom_player = Agent(
    model,
    instructions=INSTRUCTIONS,
    toolsets=[doom_mcp],
    capabilities=[hooks],
)

commander = Agent(
    model,
    instructions=COMMANDER_INSTRUCTIONS,
    capabilities=[hooks],
)

PLAYER_LIMITS = UsageLimits(request_limit=5)
COMMANDER_LIMITS = UsageLimits(request_limit=3)

def _panel(text: str | None, title: str, border: str) -> Panel:
    return Panel(escape(text or "(no output)"), title=title, border_style=border)

async def main():
    turn = 0
    commander_history = []
    player_output = ""

    try:
        result = await doom_player.run(
            'Start playing! call the start_game tool',
            usage_limits=PLAYER_LIMITS,
        )
        player_output = result.output or ""
    except UsageLimitExceeded as e:
        player_output = f"(limit reached: {e})"
    console.print(_panel(player_output, "[bold green]PLAYER[/bold green]", "green"))

    while turn < 30:
        try:
            command = await commander.run(
                f'Player report: {player_output}',
                message_history=commander_history,
                usage_limits=COMMANDER_LIMITS,
            )
            commander_history = command.all_messages()
            commander_output = command.output or ""
        except UsageLimitExceeded as e:
            commander_output = f"(limit reached: {e})"
        console.print(_panel(commander_output, "[bold magenta]COMMANDER[/bold magenta]", "magenta"))

        try:
            result = await doom_player.run(
                f'Commander instruction: {commander_output}',
                usage_limits=PLAYER_LIMITS,
            )
            player_output = result.output or ""
        except UsageLimitExceeded as e:
            player_output = f"(limit reached: {e})"
        console.print(_panel(player_output, f"[bold green]PLAYER — turn {turn + 1}[/bold green]", "green"))
        turn += 1

    console.print(Panel("Out of turns. Bye!", border_style="dim"))

if __name__ == "__main__":
    asyncio.run(main())

tunnel.stop()
