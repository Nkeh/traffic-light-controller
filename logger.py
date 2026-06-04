# =============================================================================
# logger.py — Simulation Logger
# Writes every decision cycle to a CSV file and tracks performance metrics.
# Produces summary statistics and comparison data for Phase 5 evaluation.
# =============================================================================

import csv
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque

from config import (
    LOG_FILE, LOG_HEADERS,
    Phase, TrafficProfile,
    DIRECTIONS, DIRECTION_NAMES,
)
from agent import AgentState


# ── Per-cycle log entry ───────────────────────────────────────────────────────

@dataclass
class CycleEntry:
    """One row of logged data — every field maps to a CSV column."""
    cycle:            int
    timestamp_s:      float
    profile:          str
    phase:            str
    queue_N:          int
    queue_S:          int
    queue_E:          int
    queue_W:          int
    wait_N:           float
    wait_S:           float
    wait_E:           float
    wait_W:           float
    utility_NS:       float
    utility_EW:       float
    decision:         str
    vehicles_cleared: int
    avg_wait_time:    float
    emergency_active: bool
    ped_request_N:    bool
    ped_request_S:    bool
    ped_request_E:    bool
    ped_request_W:    bool

    def to_row(self) -> list:
        """Convert to a flat list matching LOG_HEADERS order."""
        return [
            self.cycle, round(self.timestamp_s, 2), self.profile,
            self.phase,
            self.queue_N, self.queue_S, self.queue_E, self.queue_W,
            round(self.wait_N, 1), round(self.wait_S, 1),
            round(self.wait_E, 1), round(self.wait_W, 1),
            round(self.utility_NS, 2), round(self.utility_EW, 2),
            self.decision,
            self.vehicles_cleared,
            round(self.avg_wait_time, 3),
            int(self.emergency_active),
            int(self.ped_request_N), int(self.ped_request_S),
            int(self.ped_request_E), int(self.ped_request_W),
        ]


# ── Phase transition record ───────────────────────────────────────────────────

@dataclass
class PhaseTransition:
    """Records every time the signal phase changes."""
    cycle:      int
    from_phase: str
    to_phase:   str
    reason:     str
    duration_s: int    # how long the previous phase lasted


# ── Performance snapshot ──────────────────────────────────────────────────────

@dataclass
class PerformanceSnapshot:
    """
    Aggregated performance metrics over a time window.
    Used for the Phase 5 comparison report.
    """
    window_start:    int
    window_end:      int
    avg_wait_time:   float
    max_wait_time:   float
    total_cleared:   float
    throughput_vps:  float    # vehicles per second
    phase_dist:      Dict[str, float]  = field(default_factory=dict)
    emergency_count: int      = 0
    ped_served:      int      = 0


# ── Main logger ───────────────────────────────────────────────────────────────

class SimulationLogger:
    """
    Logs every decision cycle to CSV and tracks performance metrics.

    Usage:
        logger = SimulationLogger()
        logger.log_cycle(agent.state, simulator, cycle)
        ...
        logger.close()
        summary = logger.get_summary()
    """

    SNAPSHOT_INTERVAL = 100   # cycles between performance snapshots

    def __init__(
        self,
        filepath:      str  = LOG_FILE,
        console_every: int  = 10,     # print to console every N cycles
        silent:        bool = False,  # suppress all console output
    ):
        self._filepath      = filepath
        self._console_every = console_every
        self._silent        = silent

        # State tracking
        self._cycle_entries:    List[CycleEntry]      = []
        self._transitions:      List[PhaseTransition] = []
        self._snapshots:        List[PerformanceSnapshot] = []
        self._prev_phase:       Optional[Phase]       = None
        self._phase_start_cycle: int                  = 0
        self._phase_counts:     Dict[str, int]        = {p.name: 0 for p in Phase}
        self._wait_history:     deque                 = deque(maxlen=1000)
        self._cleared_history:  deque                 = deque(maxlen=1000)
        self._emergency_count:  int                   = 0
        self._ped_served_count: int                   = 0
        self._start_wall_time:  float                 = time.time()

        # Open CSV and write header
        self._file   = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(LOG_HEADERS)

        if not silent:
            print(f"[Logger] Writing to: {os.path.abspath(filepath)}")
            print(f"[Logger] Columns: {len(LOG_HEADERS)}")

    # =========================================================================
    # Primary log call — once per cycle
    # =========================================================================

    def log_cycle(
        self,
        state:            AgentState,
        timestamp_s:      float,
        vehicles_cleared: int,
        u_ns:             float,
        u_ew:             float,
    ):
        """
        Log one complete decision cycle.
        Called by main.py after agent.run_cycle() and simulator.step().
        """
        s = state

        # Build entry
        entry = CycleEntry(
            cycle            = s.cycle_count,
            timestamp_s      = timestamp_s,
            profile          = s.traffic_profile.value,
            phase            = s.current_phase.name,
            queue_N          = s.queue_length[0],
            queue_S          = s.queue_length[1],
            queue_E          = s.queue_length[2],
            queue_W          = s.queue_length[3],
            wait_N           = s.wait_time[0],
            wait_S           = s.wait_time[1],
            wait_E           = s.wait_time[2],
            wait_W           = s.wait_time[3],
            utility_NS       = u_ns,
            utility_EW       = u_ew,
            decision         = s.last_decision,
            vehicles_cleared = vehicles_cleared,
            avg_wait_time    = s.avg_wait_time,
            emergency_active = s.emergency_active,
            ped_request_N    = s.pedestrian_request[0],
            ped_request_S    = s.pedestrian_request[1],
            ped_request_E    = s.pedestrian_request[2],
            ped_request_W    = s.pedestrian_request[3],
        )

        # Write to CSV
        self._writer.writerow(entry.to_row())
        self._cycle_entries.append(entry)

        # Track phase transitions
        self._track_transition(s.current_phase, s.cycle_count, s.last_decision)

        # Update metrics
        self._phase_counts[s.current_phase.name] += 1
        self._wait_history.append(s.avg_wait_time)
        self._cleared_history.append(vehicles_cleared)
        if s.emergency_active:
            self._emergency_count += 1

        # Periodic snapshot — fires after every SNAPSHOT_INTERVAL entries
        if len(self._cycle_entries) % self.SNAPSHOT_INTERVAL == 0 and len(self._cycle_entries) > 0:
            self._take_snapshot(s.cycle_count)

        # Console output
        if not self._silent and s.cycle_count % self._console_every == 0:
            self._print_cycle(entry)

    # =========================================================================
    # Phase transition tracking
    # =========================================================================

    def _track_transition(self, phase: Phase, cycle: int, reason: str):
        """Record phase changes with duration of previous phase."""
        if self._prev_phase is None:
            self._prev_phase       = phase
            self._phase_start_cycle = cycle
            return

        if phase != self._prev_phase:
            duration = cycle - self._phase_start_cycle
            self._transitions.append(PhaseTransition(
                cycle      = cycle,
                from_phase = self._prev_phase.name,
                to_phase   = phase.name,
                reason     = reason,
                duration_s = duration,
            ))
            self._prev_phase        = phase
            self._phase_start_cycle = cycle

    # =========================================================================
    # Performance snapshots — for Phase 5 evaluation charts
    # =========================================================================

    def _take_snapshot(self, cycle: int):
        """Aggregate metrics over the last SNAPSHOT_INTERVAL cycles."""
        window = self._cycle_entries[-self.SNAPSHOT_INTERVAL:]
        if not window:
            return

        waits       = [e.avg_wait_time for e in window]
        all_waits   = [e.wait_N + e.wait_S + e.wait_E + e.wait_W for e in window]
        cleared_sum = sum(e.vehicles_cleared for e in window)
        total_secs  = len(window)

        # Phase distribution
        phase_counts = {}
        for e in window:
            phase_counts[e.phase] = phase_counts.get(e.phase, 0) + 1
        phase_dist = {k: round(v / total_secs * 100, 1) for k, v in phase_counts.items()}

        self._snapshots.append(PerformanceSnapshot(
            window_start   = cycle - self.SNAPSHOT_INTERVAL,
            window_end     = cycle,
            avg_wait_time  = round(sum(waits) / len(waits), 3),
            max_wait_time  = round(max(all_waits), 1),
            total_cleared  = cleared_sum,
            throughput_vps = round(cleared_sum / total_secs, 3),
            phase_dist     = phase_dist,
            emergency_count= sum(1 for e in window if e.emergency_active),
            ped_served     = 0,  # tracked separately if needed
        ))

    # =========================================================================
    # Summary statistics
    # =========================================================================

    def get_summary(self) -> dict:
        """
        Returns a dict of summary statistics for the full simulation run.
        Used by main.py to print the final report and feed into charts.
        """
        if not self._cycle_entries:
            return {}

        all_waits   = [e.avg_wait_time   for e in self._cycle_entries]
        all_cleared = [e.vehicles_cleared for e in self._cycle_entries]
        total_cycles = len(self._cycle_entries)
        wall_elapsed = round(time.time() - self._start_wall_time, 2)

        # Phase distribution as percentages
        total_phase_cycles = sum(self._phase_counts.values())
        phase_dist = {
            k: round(v / total_phase_cycles * 100, 1)
            for k, v in self._phase_counts.items() if v > 0
        }

        # Average phase durations from transitions
        phase_durations: Dict[str, List[int]] = {}
        for t in self._transitions:
            phase_durations.setdefault(t.from_phase, []).append(t.duration_s)
        avg_durations = {
            k: round(sum(v) / len(v), 1)
            for k, v in phase_durations.items()
        }

        return {
            "total_cycles":        total_cycles,
            "simulated_seconds":   self._cycle_entries[-1].timestamp_s,
            "wall_time_s":         wall_elapsed,
            "avg_wait_time":       round(sum(all_waits) / len(all_waits), 3),
            "min_wait_time":       round(min(all_waits), 3),
            "max_wait_time":       round(max(all_waits), 3),
            "total_vehicles_cleared": sum(all_cleared),
            "avg_throughput_vps":  round(sum(all_cleared) / total_cycles, 4),
            "phase_distribution":  phase_dist,
            "avg_phase_durations": avg_durations,
            "total_transitions":   len(self._transitions),
            "emergency_cycles":    self._emergency_count,
            "log_filepath":        os.path.abspath(self._filepath),
            "snapshots":           len(self._snapshots),
        }

    def get_snapshots(self) -> List[PerformanceSnapshot]:
        """Return all periodic snapshots — used by the chart generator."""
        return self._snapshots

    def get_transitions(self) -> List[PhaseTransition]:
        """Return all phase transitions — used for FSM analysis."""
        return self._transitions

    def get_entries(self) -> List[CycleEntry]:
        """Return all cycle entries — used for detailed chart generation."""
        return self._cycle_entries

    # =========================================================================
    # Console output
    # =========================================================================

    def _print_cycle(self, e: CycleEntry):
        """Compact one-line console output per logged cycle."""
        emrg = " [EMRG]" if e.emergency_active else ""
        print(
            f"[{e.cycle:>5}] {e.phase:<12} | "
            f"Q N:{e.queue_N:<2} S:{e.queue_S:<2} E:{e.queue_E:<2} W:{e.queue_W:<2} | "
            f"W N:{e.wait_N:<4.0f} S:{e.wait_S:<4.0f} E:{e.wait_E:<4.0f} W:{e.wait_W:<4.0f} | "
            f"U NS:{e.utility_NS:>8.1f} EW:{e.utility_EW:>8.1f} | "
            f"AvgW:{e.avg_wait_time:>5.2f}s{emrg}"
        )

    def print_summary(self):
        """Print a formatted summary table to console."""
        s = self.get_summary()
        if not s:
            print("[Logger] No data to summarise.")
            return
        divider = "=" * 60
        print(divider)
        print("  SIMULATION SUMMARY")
        print(divider)
        print(f"  Total cycles simulated : {s['total_cycles']}")
        print(f"  Simulated time         : {s['simulated_seconds']:.0f}s "
              f"({s['simulated_seconds']/3600:.2f}h)")
        print(f"  Wall clock time        : {s['wall_time_s']}s")
        print(f"  Vehicles cleared       : {s['total_vehicles_cleared']}")
        print(f"  Avg throughput         : {s['avg_throughput_vps']} veh/s")
        print(divider)
        print(f"  Avg wait time          : {s['avg_wait_time']}s")
        print(f"  Min wait time          : {s['min_wait_time']}s")
        print(f"  Max wait time          : {s['max_wait_time']}s")
        print(divider)
        print("  Phase distribution:")
        for phase, pct in sorted(s['phase_distribution'].items()):
            bar = "|" * int(pct / 2)
            print(f"    {phase:<14} {pct:>5.1f}%  {bar}")
        print(divider)
        print("  Avg phase durations:")
        for phase, dur in sorted(s['avg_phase_durations'].items()):
            print(f"    {phase:<14} {dur:>5.1f}s avg")
        print(divider)
        print(f"  Phase transitions      : {s['total_transitions']}")
        print(f"  Emergency cycles       : {s['emergency_cycles']}")
        print(f"  Log file               : {s['log_filepath']}")
        print(divider)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self):
        """Flush and close the CSV file."""
        self._file.flush()
        self._file.close()
        if not self._silent:
            print(f"[Logger] Closed. {len(self._cycle_entries)} rows written.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()