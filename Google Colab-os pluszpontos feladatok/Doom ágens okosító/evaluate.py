"""Evaluation harness for Doom MCP GameManager.

Runs a simple baseline agent (explore, fight) for N episodes and records metrics.
"""

import argparse
import json
import time
from pathlib import Path

import vizdoom
import shutil
import tempfile
import os

# Copy vizdoom scenarios to a temporary ASCII-safe directory to avoid
# Unicode decode issues when loading .cfg files on some platforms.
src_scenarios = vizdoom.scenarios_path
tmpdir = tempfile.mkdtemp(prefix="vizsc_")
shutil.copytree(src_scenarios, tmpdir, dirs_exist_ok=True)
vizdoom.scenarios_path = tmpdir

from doom_mcp.game_manager import GameManager
from pydantic_ai import Agent
import asyncio
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.toolsets.fastmcp import FastMCPToolset


INSTRUCTIONS = """
You are an autonomous Doom player controlled turn-by-turn by a commander.
Each turn you receive ONE objective. Execute it with the minimum tool calls needed, then stop and report.
You MUST use your tools — do not describe actions, execute them.

Startup: If no game is running, call `start_game` with map MAP02.

One task per turn: Execute the objective given. Stop after completing it.

After every turn write a short REPORT and NEXT objective line.
"""


def ensure_doom_mcp_server(port: int = 8001, timeout: float = 30.0) -> None:
    import urllib.request
    import subprocess
    import os
    import sys
    import time
    from pathlib import Path

    url = f"http://127.0.0.1:{port}/sse"
    def ready():
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            return False

    if ready():
        return

    # start server using fastmcp
    project_root = Path(__file__).resolve().parent / "tetsuo-doom"
    venv_fastmcp = Path(sys.executable).with_name("fastmcp.exe")
    if venv_fastmcp.exists():
        fastmcp_executable = str(venv_fastmcp)
    else:
        fastmcp_executable = shutil.which("fastmcp")
    if fastmcp_executable is None:
        raise RuntimeError("fastmcp not found; start the doom MCP server manually")

    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.Popen([fastmcp_executable, "run", "src/doom_mcp/server.py", "--transport", "sse", "--port", str(port)], cwd=str(project_root), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    start = time.time()
    while time.time() - start < timeout:
        if ready():
            return
        if proc.poll() is not None:
            raise RuntimeError("Doom MCP server exited unexpectedly")
        time.sleep(0.5)
    raise RuntimeError("Doom MCP server did not become ready in time")


def evaluate_with_agent(episodes: int, scenario: str, seed: int, out_path: str, window_visible: bool = False):
    # Ensure MCP server is running
    ensure_doom_mcp_server()
    mcp_url = f"http://localhost:8001/sse"

    # Attempt to construct an Ollama model; if not available, raise and fallback
    try:
        provider = OllamaProvider(base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'))
        model_name = os.getenv('OLLAMA_MODEL', 'qwen3-coder-next:cloud')
        model = OllamaModel(model_name, provider=provider)
    except Exception as e:
        raise RuntimeError(f"Could not initialize Ollama model: {e}")

    doom_mcp = FastMCPToolset(mcp_url, max_retries=3)
    agent = Agent(model, instructions=INSTRUCTIONS, toolsets=[doom_mcp])

    results = []
    async def run_single_episode(i, s, window_visible: bool = False):
        print(f"Agent run {i+1}/{episodes} seed={s}")
        output_summary = {
            'scenario': scenario,
            'seed': s,
            'agent_turns': [],
        }

        # Start the game via the agent
        # Ensure a visible window if requested by calling the start_game tool directly
        if window_visible:
            try:
                maybe = doom_mcp.call_tool("start_game", scenario=scenario, window_visible=True)
                if asyncio.iscoroutine(maybe):
                    await maybe
                print("Called start_game with window_visible=True via toolset")
            except Exception as e:
                print("Warning: could not call start_game via toolset:", e)

        res = await agent.run('Start the game and begin play.')
        out = getattr(res, 'output', '')
        print('Initial agent output:', (out or '')[:200])
        output_summary['agent_turns'].append(out)

        # Do a few turns: instruct agent to explore
        turns = 3
        for t in range(turns):
            cmd = 'Commander instruction: Explore the area.'
            r = await agent.run(cmd)
            out = getattr(r, 'output', '')
            print(f"Turn {t+1} output:", (out or '')[:200])
            output_summary['agent_turns'].append(out)

        return output_summary

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        for i in range(episodes):
            s = seed + i
            try:
                summary = loop.run_until_complete(run_single_episode(i, s, window_visible=window_visible))
                results.append(summary)
            except Exception:
                raise
    finally:
        loop.close()

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'Agent evaluation wrote {len(results)} results to {out_path}')


def run_episode(manager: GameManager, scenario: str, seed: int, max_cycles: int = 20, window_visible: bool = False) -> dict:
    manager.start(scenario=scenario, seed=seed, window_visible=window_visible)

    episode_start = time.time()
    cycles = 0
    last_state = None

    try:
        while cycles < max_cycles:
            # Get current state
            state = manager.get_state()
            last_state = state
            if state.get("episode_finished"):
                break

            # Explore a short amount; stop when enemy/item seen
            res = manager.explore(max_tics=100, stop_on_enemy=True, stop_on_item=True)
            summary = res.get("action_summary", {})

            stop_reason = summary.get("stop_reason")
            # If enemy spotted, assess and shoot the top threat
            if stop_reason == "enemy_spotted":
                ta = manager.get_threat_assessment()
                threats = ta.get("threats", [])
                if threats:
                    target_id = threats[0].get("id")
                    manager.aim_and_shoot(target_id, shots=3, max_tics=150)

            cycles += 1

            # small sleep to avoid busy loop (GameManager runs fast)
            time.sleep(0.05)

        # final state
        state = manager.get_state()
        total_reward = state.get("total_reward")
        vars = state.get("game_variables", {})
        result = {
            "scenario": scenario,
            "seed": seed,
            "episode_finished": state.get("episode_finished", False),
            "total_reward": total_reward,
            "kills": vars.get("KILLCOUNT"),
            "health": vars.get("HEALTH"),
            "tics": state.get("tic"),
            "cycles": cycles,
            "duration_s": time.time() - episode_start,
        }
        return result

    finally:
        # Ensure the manager is stopped between episodes
        try:
            manager.stop()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-n", type=int, default=3)
    parser.add_argument("--scenario", "-s", type=str, default="basic")
    parser.add_argument("--out", "-o", type=str, default="results.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-agent", action="store_true", help="Use the pydantic-ai doom_player agent (requires Ollama) ")
    parser.add_argument("--window-visible", action="store_true", help="Open a visible ViZDoom window during episodes")
    args = parser.parse_args()

    out_path = Path(args.out)
    results = []
    manager = GameManager()

    if args.use_agent:
        try:
            evaluate_with_agent(args.episodes, args.scenario, args.seed, args.out, window_visible=args.window_visible)
            return
        except Exception as e:
            print('Agent evaluation failed, falling back to baseline:', e)

    for i in range(args.episodes):
        seed = args.seed + i
        print(f"Running episode {i+1}/{args.episodes} scenario={args.scenario} seed={seed}")
        r = run_episode(manager, args.scenario, seed, window_visible=args.window_visible)
        print(" ->", r)
        results.append(r)

    # write JSON lines
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    print(f"Wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
