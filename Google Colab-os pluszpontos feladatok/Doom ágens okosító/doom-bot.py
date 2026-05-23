import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    SSHTunnelForwarder = None

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel

console = Console()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, show_path=False, markup=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("doom")

from dataclasses import replace

from pydantic_ai import Agent, ModelRequestContext, ModelResponse, RunContext, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded, ModelHTTPError
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPToolset
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



# The SSH tunnel is only needed for remote resources. Keep it opt-in so the
# local Ollama workflow does not depend on sshtunnel/paramiko compatibility.
tunnel = None
if os.getenv("ENABLE_SSH_TUNNEL", "0") == "1":
    if SSHTunnelForwarder is None:
        raise RuntimeError("ENABLE_SSH_TUNNEL=1 requires the sshtunnel package to be installed.")
    try:
        tunnel = SSHTunnelForwarder(
            ("spark.mit.bme.hu", 10222),
            ssh_username="uname",
            ssh_pkey="/Users/uname/.ssh/id_rsa",
            ssh_private_key_password=os.getenv("SSH_KEY_PASSWORD"),
            remote_bind_address=("localhost", 8000),
            local_bind_address=("localhost", 8000),
        )
        tunnel.start()
    except Exception as exc:
        log.warning("SSH tunnel startup failed; continuing without it: %s", exc)
# Use Ollama by default. Ollama should be running locally or reachable via OLLAMA_BASE_URL.
# Adjust model name to one available in your Ollama instance via OLLAMA_MODEL.
# Example models present locally: 'qwen3.5:2b-q4_K_M', 'qwen3.5:cloud', 'qwen3-coder-next:cloud'.
ollama_provider = OllamaProvider(
    base_url=os.getenv('OLLAMA_BASE_URL', "http://localhost:11434/v1"),
)
model = OllamaModel(os.getenv('OLLAMA_MODEL', 'qwen3-coder-next:cloud'), provider=ollama_provider)

# Print connection info (avoid calling functions not yet defined)
_doom_port = os.getenv("DOOM_MCP_PORT", "8001")
_doom_url = os.getenv("DOOM_MCP_URL", f"http://localhost:{_doom_port}/sse")
_mem_port = os.getenv("MEMORY_MCP_PORT", "8002")
_mem_url = os.getenv("MEMORY_MCP_URL", f"http://localhost:{_mem_port}/sse")
log.info("DOOM_MCP_URL=%s", _doom_url)
log.info("MEMORY_MCP_URL=%s", _mem_url)
log.info("OLLAMA_BASE_URL=%s", os.getenv('OLLAMA_BASE_URL', 'not set'))
log.info("OLLAMA_MODEL=%s", os.getenv('OLLAMA_MODEL', 'not set'))


def _server_url() -> str:
    port = os.getenv("DOOM_MCP_PORT", "8001")
    return os.getenv("DOOM_MCP_URL", f"http://localhost:{port}/sse")


def _server_is_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _start_doom_mcp_server() -> subprocess.Popen:
    project_root = Path(__file__).resolve().parent / "tetsuo-doom"
    if not project_root.exists():
        raise RuntimeError(f"Cannot find tetsuo-doom backend at {project_root}")

    venv_fastmcp = Path(sys.executable).with_name("fastmcp.exe")
    if venv_fastmcp.exists():
        fastmcp_executable = str(venv_fastmcp)
    else:
        fastmcp_executable = shutil.which("fastmcp")
    if fastmcp_executable is None:
        raise RuntimeError("fastmcp is required to auto-start the Doom MCP server, but it was not found on PATH or in the active virtual environment.")

    try:
        import fastmcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "fastmcp is not available in the current Python environment; install the project dependencies first."
        ) from exc

    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    port = os.getenv("DOOM_MCP_PORT", "8001")

    return subprocess.Popen(
        [fastmcp_executable, "run", "src/doom_mcp/server.py", "--transport", "sse", "--port", port],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _tail_text(lines: str | bytes, limit: int = 20) -> str:
    if isinstance(lines, bytes):
        lines = lines.decode("utf-8", errors="replace")
    parts = [line.rstrip() for line in lines.splitlines() if line.strip()]
    if not parts:
        return ""
    return "\n".join(parts[-limit:])


def ensure_doom_mcp_server(url: str | None = None) -> None:
    target_url = url or _server_url()
    if _server_is_ready(target_url):
        return

    if os.getenv("AUTO_START_DOOM_MCP", "1") != "1":
        raise RuntimeError(
            f"Doom MCP server is not reachable at {target_url}. Start it manually or set AUTO_START_DOOM_MCP=1."
        )

    proc = _start_doom_mcp_server()

    for _ in range(30):
        if _server_is_ready(target_url, timeout=1.5):
            return
        if proc.poll() is not None:
            stderr_output = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(
                f"Doom MCP server exited before becoming ready at {target_url}.\n"
                f"Backend stderr:\n{_tail_text(stderr_output)}"
            )
        time.sleep(0.5)

    stderr_output = proc.stderr.read() if proc.stderr is not None else ""
    raise RuntimeError(
        f"Doom MCP server did not become ready at {target_url}.\n"
        f"Backend stderr:\n{_tail_text(stderr_output)}"
    )


def _memory_server_url() -> str:
    port = os.getenv("MEMORY_MCP_PORT", "8002")
    return os.getenv("MEMORY_MCP_URL", f"http://localhost:{port}/sse")


def _start_memory_mcp_server() -> subprocess.Popen:
    project_root = Path(__file__).resolve().parent / "tetsuo-doom"
    if not project_root.exists():
        raise RuntimeError(f"Cannot find tetsuo-doom backend at {project_root}")

    venv_fastmcp = Path(sys.executable).with_name("fastmcp.exe")
    if venv_fastmcp.exists():
        fastmcp_executable = str(venv_fastmcp)
    else:
        fastmcp_executable = shutil.which("fastmcp")
    if fastmcp_executable is None:
        raise RuntimeError("fastmcp is required to auto-start the Memory MCP server, but it was not found on PATH or in the active virtual environment.")

    try:
        import fastmcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "fastmcp is not available in the current Python environment; install the project dependencies first."
        ) from exc

    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    port = os.getenv("MEMORY_MCP_PORT", "8002")

    return subprocess.Popen(
        [fastmcp_executable, "run", "src/doom_mcp/memory_server.py", "--transport", "sse", "--port", port],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def ensure_memory_mcp_server(url: str | None = None) -> None:
    target_url = url or _memory_server_url()
    if _server_is_ready(target_url):
        return

    if os.getenv("AUTO_START_MEMORY_MCP", "1") != "1":
        raise RuntimeError(
            f"Memory MCP server is not reachable at {target_url}. Start it manually or set AUTO_START_MEMORY_MCP=1."
        )

    proc = _start_memory_mcp_server()

    for _ in range(30):
        if _server_is_ready(target_url, timeout=1.5):
            return
        if proc.poll() is not None:
            stderr_output = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(
                f"Memory MCP server exited before becoming ready at {target_url}.\n"
                f"Backend stderr:\n{_tail_text(stderr_output)}"
            )
        time.sleep(0.5)

    stderr_output = proc.stderr.read() if proc.stderr is not None else ""
    raise RuntimeError(
        f"Memory MCP server did not become ready at {target_url}.\n"
        f"Backend stderr:\n{_tail_text(stderr_output)}"
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
If no game is running, start a new Doom II game on MAP02 with difficulty 3 and a visible window if available.
Call `start_game` alone and wait for it to finish before doing anything else.
Do not call `start_game` again unless the game crashes.

## One task per turn
Execute exactly the objective given by the commander. Do not chain multiple objectives.
- If told to explore, call the `explore` tool once.
- If told to fight an enemy, call `get_threat_assessment` first, then `aim_and_shoot` or `strafe_and_shoot` once.
- If told to collect an item, call `move_to` once with the target object id.
- If told to open a door or use a switch, call `move_to` once with the target object id and use enabled.
Stop after completing the task. Do not keep exploring or fighting after the objective is done.

## Exploration
- Use `explore` with stop options to stop on enemies or items.
- If `explore` returns stuck or max tics, call `get_navigation_info` and continue exploring.

## Combat
- `get_threat_assessment` returns enemy IDs and priorities.
- `strafe_and_shoot` is for hitscan enemies such as chaingunners and former humans.
- `aim_and_shoot` is for other enemies.
- `retreat` moves away when health is critical.

## Doors and keys
Doom doors are opened by walking up and using them. Keys unlock color-coded locked doors.
- Call `move_to` with use enabled for doors and switches.
- Keys are visible in `get_objects` with type `key`.
- If you spot a key or locked door, report its object id and color so the commander can plan.

## Items and switches
- Collect items by calling `move_to` with the target object id.
- Activate switches by calling `move_to` with use enabled.

## Exit
- Scan `get_objects` for names containing exit, switch, or teleport.
- Call `move_to` with use enabled to finish the level.
- When `episode_finished` is true in any result, call `new_episode` to advance to the next map.

## Rules
- Always call exactly one tool or a short tool sequence per turn.
- Never call `stop_game`.
- Never call `take_action` directly.
- Stop after completing the objective.

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
Health below 40: prioritize retreat and then explore for health.
- Active enemies spotted: prioritize aim_and_shoot or strafe_and_shoot.
No enemies, healthy: direct the player to explore for enemies.
Exit or switch found: direct move_to to activate it.
- Stuck / max_tics: direct get_navigation_info then explore in a different direction.
- episode_finished seen: direct new_episode immediately.

## How to give instructions
Respond in exactly this format:

  OBJECTIVE: <what to do>
  REASON: <why, based on the player's report>
    HINT: Call the relevant tool with the target object id or boolean options.

Examples:
  OBJECTIVE: Hunt for enemies.
  REASON: Area is clear and exploration progress is low.
    HINT: Call `explore`, then use `aim_and_shoot` with the enemy id from `get_threat_assessment`.

  OBJECTIVE: Activate the exit switch.
  REASON: No enemies remain and you can see an exit switch in get_objects.
    HINT: Call `move_to` with the switch id and use enabled.

  OBJECTIVE: Find health — HP is critical.
  REASON: Health is below 30.
    HINT: Call `retreat`, then `explore`; when an item is found, call `move_to` with its object id.

Keep instructions short and actionable. One objective per turn.
"""

ensure_doom_mcp_server()
doom_mcp = MCPToolset(_server_url(), max_retries=3)
ensure_memory_mcp_server()
memory_mcp = MCPToolset(os.getenv("MEMORY_MCP_URL", "http://localhost:8002/sse"), max_retries=3)
doom_player = Agent(
    model,
    instructions=INSTRUCTIONS,
    toolsets=[doom_mcp, memory_mcp],
    capabilities=[hooks],
)

commander = Agent(
    model,
    instructions=COMMANDER_INSTRUCTIONS,
    capabilities=[hooks],
)

PLAYER_LIMITS = UsageLimits(request_limit=20)
COMMANDER_LIMITS = UsageLimits(request_limit=10)

def _panel(text: str | None, title: str, border: str) -> Panel:
    return Panel(escape(text or "(no output)"), title=title, border_style=border)

async def main():
    turn = 0
    commander_history = []
    player_output = ""

    try:
        result = await doom_player.run(
            'Start the game and begin play.',
            usage_limits=PLAYER_LIMITS,
        )
        player_output = result.output or ""
    except UsageLimitExceeded as e:
        player_output = f"(limit reached: {e})"
    except ModelHTTPError as e:
        # Rate limit or other HTTP model error: stop the game and exit
        await _handle_model_http_error(e)
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
        except ModelHTTPError as e:
            await _handle_model_http_error(e)
        console.print(_panel(player_output, f"[bold green]PLAYER — turn {turn + 1}[/bold green]", "green"))
        turn += 1

    console.print(Panel("Out of turns. Bye!", border_style="dim"))


async def _handle_model_http_error(err: Exception) -> None:
    """Attempt to stop the running game, print the error, and exit.

    This tries a few fallback ways to invoke the `stop_game` tool on the
    Doom MCP server. If all attempts fail, the error is still printed and
    the process exits so the operator can intervene.
    """
    err_msg = f"Model HTTP error / rate limit: {err}"
    log.error(err_msg)
    console.print(Panel(err_msg, title="[red]MODEL ERROR[/red]", border_style="red"))

    # Try toolset-level calls (best-effort; different fastmcp client versions vary)
    try:
        if hasattr(doom_mcp, "call_tool"):
            maybe = doom_mcp.call_tool("stop_game")
            if asyncio.iscoroutine(maybe):
                await maybe
            log.info("Requested stop_game via doom_mcp.call_tool")
            return
        if hasattr(doom_mcp, "call_tool_sync"):
            try:
                doom_mcp.call_tool_sync("stop_game")
                log.info("Requested stop_game via doom_mcp.call_tool_sync")
                return
            except Exception:
                pass
        # Some toolset proxies expose tools as attributes
        if hasattr(doom_mcp, "stop_game") and callable(getattr(doom_mcp, "stop_game")):
            maybe = getattr(doom_mcp, "stop_game")()
            if asyncio.iscoroutine(maybe):
                await maybe
            log.info("Requested stop_game via doom_mcp.stop_game attribute")
            return
    except Exception as exc:
        log.warning("Best-effort stop_game attempt failed: %s", exc)

    # Last resort: try a simple HTTP POST to an endpoint (best-guess)
    try:
        srv = _server_url().replace("/sse", "")
        url = f"{srv}/tools/stop_game"
        req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=2)
        log.info("Requested stop_game via HTTP POST %s", url)
        return
    except Exception as exc:
        log.warning("HTTP stop_game attempt failed: %s", exc)

    # If we couldn't stop programmatically, still exit so operator can intervene
    log.error("Could not automatically stop the game. Please stop the Doom process manually.")
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

if tunnel is not None:
    tunnel.stop()
