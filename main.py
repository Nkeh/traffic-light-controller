# =============================================================================
# main.py — Entry point
# Wires simulator + agent + logger together and runs the simulation loop.
# Can run in two modes:
#   headless  — console output + CSV log only (no GUI)
#   gui       — full Pygame visual simulation (default)
#
# Usage:
#   python main.py              — GUI mode
#   python main.py --headless   — headless mode
#   python main.py --headless --cycles 500
#   python main.py --headless --profile MORNING_RUSH
#   python main.py --headless --seed 123
# =============================================================================

import argparse
import sys
import time

from config import (
    Phase, TrafficProfile,
    INITIAL_PHASE, INITIAL_PROFILE,
    CYCLE_DURATION_MS, LOG_FILE,
    NORTH, SOUTH, EAST, WEST,
)
from simulator import IntersectionSimulator
from agent    import TrafficLightAgent
from logger   import SimulationLogger


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Traffic Light Controller Simulation"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI (console + CSV only)"
    )
    parser.add_argument(
        "--cycles", type=int, default=500,
        help="Number of cycles to run in headless mode (default: 500)"
    )
    parser.add_argument(
        "--profile",
        choices=[p.value for p in TrafficProfile],
        default=INITIAL_PROFILE.value,
        help="Starting traffic profile"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--start-hour", type=float, default=9.0,
        help="Simulated start hour 0-24 (default: 9.0 = MIDDAY)"
    )
    parser.add_argument(
        "--log", type=str, default=LOG_FILE,
        help=f"CSV log file path (default: {LOG_FILE})"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Simulation speed multiplier for headless (default: 1.0 = real-time)"
    )
    parser.add_argument(
        "--silent", action="store_true",
        help="Suppress cycle-by-cycle console output"
    )
    return parser.parse_args()


# ── Core simulation loop ──────────────────────────────────────────────────────

def build_components(profile_str: str, seed: int, start_hour: float):
    """
    Instantiate and return the three core components.
    Called by both headless and GUI modes.
    """
    profile = TrafficProfile(profile_str)

    simulator = IntersectionSimulator(
        initial_profile = profile,
        initial_phase   = INITIAL_PHASE,
        seed            = seed,
        start_hour      = start_hour,
    )
    agent = TrafficLightAgent(
        initial_phase   = INITIAL_PHASE,
        initial_profile = profile,
    )
    return simulator, agent


def run_one_cycle(simulator: IntersectionSimulator,
                  agent:     TrafficLightAgent,
                  logger:    SimulationLogger) -> Phase:
    """
    Execute one complete simulation cycle:
      1. Agent reads sensors
      2. Agent decides phase
      3. Simulator applies phase
      4. Simulator advances world by one step
      5. Logger records everything

    Returns the phase active after this cycle.
    """
    # 1 + 2 — sense and decide
    reading   = simulator.get_sensor_reading()
    new_phase = agent.run_cycle(reading)

    # 3 — apply decision to world
    simulator.apply_phase(new_phase)

    # 4 — advance world
    simulator.step()

    # 5 — log
    u_ns, u_ew = agent.utility_scores
    logger.log_cycle(
        state            = agent.state,
        timestamp_s      = float(simulator.cycle),
        vehicles_cleared = simulator.total_cleared,
        u_ns             = u_ns,
        u_ew             = u_ew,
    )

    return new_phase


# ── Headless mode ─────────────────────────────────────────────────────────────

def run_headless(args):
    """
    Run the simulation without a GUI.
    Prints cycle-by-cycle status to console and writes CSV log.
    Used for debugging, testing, and Phase 5 data collection.
    """
    print("=" * 65)
    print("  AUTONOMOUS TRAFFIC LIGHT CONTROLLER — Headless Mode")
    print("=" * 65)
    print(f"  Profile    : {args.profile}")
    print(f"  Cycles     : {args.cycles}")
    print(f"  Seed       : {args.seed}")
    print(f"  Start hour : {args.start_hour:.1f}")
    print(f"  Log file   : {args.log}")
    print("=" * 65)
    print()

    simulator, agent = build_components(args.profile, args.seed, args.start_hour)
    cycle_ms = CYCLE_DURATION_MS / max(args.speed, 0.01)

    with SimulationLogger(
        filepath      = args.log,
        console_every = 20,
        silent        = args.silent,
    ) as logger:

        start_wall = time.time()

        for cycle in range(args.cycles):
            cycle_start = time.time()

            run_one_cycle(simulator, agent, logger)

            # Real-time pacing (ignored if speed > 10 — runs as fast as possible)
            if args.speed <= 10.0:
                elapsed_ms  = (time.time() - cycle_start) * 1000
                sleep_ms    = cycle_ms - elapsed_ms
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000)

        wall_time = time.time() - start_wall
        print(f"\n  Completed {args.cycles} cycles in {wall_time:.2f}s wall time.")
        print()
        logger.print_summary()

    return simulator, agent, logger


# ── GUI mode ──────────────────────────────────────────────────────────────────

def run_gui(args):
    """
    Launch the Pygame GUI simulation.
    Imports gui.py only when needed — avoids pygame import in headless mode.
    """
    try:
        import pygame
    except ImportError:
        print("[ERROR] pygame is not installed.")
        print("        Run:  pip install pygame")
        sys.exit(1)

    from gui import TrafficGUI

    simulator, agent = build_components(args.profile, args.seed, args.start_hour)

    with SimulationLogger(
        filepath      = args.log,
        console_every = 999,    # GUI handles its own display
        silent        = True,
    ) as logger:

        gui = TrafficGUI(simulator, agent, logger)
        gui.run()

    print()
    logger.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()