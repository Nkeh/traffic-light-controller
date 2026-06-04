# =============================================================================
# simulator.py — Intersection Simulator
# Models the physical intersection: vehicle arrivals, queue management,
# pedestrian requests, emergency events, and sensor output.
# The agent never touches this file directly — it only reads sensor readings.
# =============================================================================

import random
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    Phase, TrafficProfile, DIRECTIONS, DIRECTION_NAMES,
    NORTH, SOUTH, EAST, WEST,
    ARRIVAL_RATES, SATURATION_FLOW_RATE, MAX_QUEUE_LENGTH,
    PHASE_SERVES, RANDOM_SEED,
    EMERGENCY_ALL_RED_DURATION, EMERGENCY_CLEAR_DURATION,
    INITIAL_PHASE, INITIAL_PROFILE,
)


# ── Sensor reading snapshot ───────────────────────────────────────────────────
# This is the only object the agent receives from the simulator each cycle.
# It is read-only from the agent's perspective — a clean interface boundary.

@dataclass(frozen=True)
class SensorReading:
    """
    Snapshot of all sensor data at a given cycle.
    Passed to the agent every decision cycle via simulator.get_sensor_reading().
    """
    cycle:            int
    timestamp_s:      float                  # simulated seconds elapsed

    # Vehicle sensors
    queue_length:     tuple                  # (N, S, E, W) vehicle counts
    arrival_rate:     tuple                  # (N, S, E, W) rolling avg veh/s

    # Phase & timing (what the agent's own timer sees)
    current_phase:    Phase
    phase_elapsed:    int                    # seconds in current phase

    # Override flags
    emergency_active:     bool
    emergency_direction:  Optional[int]      # NORTH/SOUTH/EAST/WEST or None
    pedestrian_request:   tuple             # (N, S, E, W) booleans

    # Time context
    hour_of_day:      float                  # 0.0 – 24.0
    traffic_profile:  TrafficProfile

    def __str__(self):
        dirs = ["N", "S", "E", "W"]
        q = " ".join(f"{dirs[i]}:{self.queue_length[i]}" for i in DIRECTIONS)
        p = " ".join(f"{dirs[i]}:{int(self.pedestrian_request[i])}" for i in DIRECTIONS)
        return (
            f"Cycle {self.cycle:>5} | {self.current_phase.name:<12} "
            f"elapsed={self.phase_elapsed:>3}s | "
            f"Queues [{q}] | Ped [{p}] | "
            f"Emrg={self.emergency_active}"
        )


# ── Vehicle queue model ───────────────────────────────────────────────────────

class LaneQueue:
    """
    Models a single approach lane: tracks vehicle count, computes rolling
    arrival rate, and handles vehicles joining and clearing each cycle.
    """

    ROLLING_WINDOW = 10   # cycles used for arrival rate smoothing

    def __init__(self, direction: int, arrival_rate: float):
        self.direction    = direction
        self.name         = DIRECTION_NAMES[direction]
        self._queue       = 0
        self._base_rate   = arrival_rate
        self._arrivals_window: List[int] = []   # recent arrivals for rolling avg

    # ── Arrivals ──────────────────────────────────────────────────────────────

    def add_arrivals(self, arrivals: int):
        """Add newly arrived vehicles, capped at MAX_QUEUE_LENGTH."""
        self._queue = min(self._queue + arrivals, MAX_QUEUE_LENGTH)
        self._arrivals_window.append(arrivals)
        if len(self._arrivals_window) > self.ROLLING_WINDOW:
            self._arrivals_window.pop(0)

    def generate_arrivals(self) -> int:
        """
        Poisson-distributed arrivals for one cycle second.
        lambda = base arrival rate, rounded to nearest int.
        Returns number of new vehicles that arrived this cycle.
        """
        lam = self._base_rate
        # Poisson approximation: sample from random — simple and readable
        arrivals = 0
        prob = lam
        while random.random() < prob and arrivals < 5:
            arrivals += 1
            prob *= lam / (arrivals + 1)   # Poisson probability mass function
        return arrivals

    # ── Clearing ──────────────────────────────────────────────────────────────

    def clear_vehicles(self) -> int:
        """
        Remove vehicles that pass through during a green phase.
        Returns number of vehicles cleared this cycle.
        Assumption A2 (all straight) + A7 (instant obedience):
        vehicles clear at SATURATION_FLOW_RATE per second of green.
        """
        cleared = min(self._queue, SATURATION_FLOW_RATE)
        self._queue = max(0, self._queue - cleared)
        return cleared

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._queue

    @property
    def rolling_arrival_rate(self) -> float:
        if not self._arrivals_window:
            return self._base_rate
        return round(sum(self._arrivals_window) / len(self._arrivals_window), 3)

    def update_base_rate(self, new_rate: float):
        """Called when traffic profile changes."""
        self._base_rate = new_rate

    def __repr__(self):
        return f"LaneQueue({self.name}, count={self._queue}, rate={self.rolling_arrival_rate})"


# ── Main simulator ────────────────────────────────────────────────────────────

class IntersectionSimulator:
    """
    Simulates the physical intersection environment.

    Responsibilities:
      - Maintain vehicle queues for all four lanes
      - Generate stochastic vehicle arrivals each cycle
      - Clear vehicles when the agent activates a green phase
      - Handle emergency vehicle events
      - Handle pedestrian button presses
      - Track simulated time and traffic profile
      - Expose sensor readings to the agent via get_sensor_reading()

    The agent calls:
      reading = simulator.get_sensor_reading()   — read world state
      simulator.apply_phase(phase)               — apply agent's decision
      simulator.step()                            — advance one cycle
    """

    def __init__(
        self,
        initial_profile: TrafficProfile = INITIAL_PROFILE,
        initial_phase:   Phase          = INITIAL_PHASE,
        seed:            Optional[int]  = RANDOM_SEED,
        start_hour:      float          = 9.0,    # simulated start time
    ):
        if seed is not None:
            random.seed(seed)

        self._profile       = initial_profile
        self._phase         = initial_phase
        self._phase_elapsed = 0
        self._cycle         = 0
        self._timestamp_s   = 0.0
        self._hour          = start_hour

        # Vehicle queues — one per direction
        rates = ARRIVAL_RATES[initial_profile]
        self._lanes = {
            NORTH: LaneQueue(NORTH, rates[NORTH]),
            SOUTH: LaneQueue(SOUTH, rates[SOUTH]),
            EAST:  LaneQueue(EAST,  rates[EAST]),
            WEST:  LaneQueue(WEST,  rates[WEST]),
        }

        # Emergency state
        self._emergency_active    = False
        self._emergency_direction: Optional[int] = None
        self._emergency_timer     = 0

        # Manual profile lock — set True by GUI to prevent auto hour-switching
        self.profile_locked = False

        # Pedestrian requests — set externally or randomly
        self._pedestrian_request = [False, False, False, False]
        self._ped_random_prob    = 0.005   # probability of random press per cycle

        # Performance tracking (simulator-side ground truth)
        self._total_cleared = 0
        self._total_arrivals = 0

    # =========================================================================
    # Public interface — called by agent and main loop
    # =========================================================================

    def step(self):
        """
        Advance the simulation by one cycle (1 second).
        Order: arrivals → clearings → pedestrian generation → time update.
        Called AFTER the agent has called apply_phase() for this cycle.
        """
        self._cycle       += 1
        self._timestamp_s += 1.0
        self._phase_elapsed += 1
        self._advance_hour()
        self._generate_arrivals()
        self._clear_vehicles()
        self._handle_emergency_timer()
        self._maybe_generate_pedestrian_request()

    def apply_phase(self, phase: Phase):
        """
        Called by the agent to apply its phase decision.
        If the phase changes, reset the phase elapsed timer.
        """
        if phase != self._phase:
            self._phase         = phase
            self._phase_elapsed = 0
            # Serve any pending pedestrian requests for this phase's crossings
            self._serve_pedestrian_requests(phase)

    def get_sensor_reading(self) -> SensorReading:
        """
        Returns a frozen snapshot of all sensor data for the current cycle.
        This is the only data the agent receives — clean interface boundary.
        """
        return SensorReading(
            cycle             = self._cycle,
            timestamp_s       = self._timestamp_s,
            queue_length      = tuple(self._lanes[d].count for d in DIRECTIONS),
            arrival_rate      = tuple(self._lanes[d].rolling_arrival_rate for d in DIRECTIONS),
            current_phase     = self._phase,
            phase_elapsed     = self._phase_elapsed,
            emergency_active  = self._emergency_active,
            emergency_direction = self._emergency_direction,
            pedestrian_request  = tuple(self._pedestrian_request),
            hour_of_day       = self._hour,
            traffic_profile   = self._profile,
        )

    # ── External triggers ─────────────────────────────────────────────────────

    def trigger_emergency(self, direction: int):
        """
        Simulates an emergency vehicle approaching from a given direction.
        Sets the emergency flag — the agent will detect this on next read.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")
        self._emergency_active    = True
        self._emergency_direction = direction
        self._emergency_timer     = 0

    def clear_emergency(self):
        """Manually clear emergency (used in testing)."""
        self._emergency_active    = False
        self._emergency_direction = None
        self._emergency_timer     = 0

    def press_pedestrian_button(self, direction: int):
        """
        Simulates a pedestrian pressing the crossing button on a given approach.
        Sets the request flag — the agent adds a delta bonus to that phase.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")
        self._pedestrian_request[direction] = True

    def set_profile(self, profile: TrafficProfile, lock: bool = False):
        """
        Change the traffic profile mid-simulation.
        Pass lock=True from the GUI to prevent auto hour-switching.
        """
        self._profile = profile
        if lock:
            self.profile_locked = True
        rates = ARRIVAL_RATES[profile]
        for d in DIRECTIONS:
            self._lanes[d].update_base_rate(rates[d])

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def phase_elapsed(self) -> int:
        return self._phase_elapsed

    @property
    def profile(self) -> TrafficProfile:
        return self._profile

    @property
    def total_cleared(self) -> int:
        return self._total_cleared

    @property
    def total_arrivals(self) -> int:
        return self._total_arrivals

    @property
    def emergency_active(self) -> bool:
        return self._emergency_active

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _generate_arrivals(self):
        """Generate stochastic vehicle arrivals for all lanes this cycle."""
        for d in DIRECTIONS:
            arrivals = self._lanes[d].generate_arrivals()
            self._lanes[d].add_arrivals(arrivals)
            self._total_arrivals += arrivals

    def _clear_vehicles(self):
        """
        Clear vehicles from lanes that are currently green.
        Only lanes served by the active green phase clear this cycle.
        Assumption A2 (straight only) + A7 (instant obedience) apply here.
        """
        if self._phase not in PHASE_SERVES:
            return   # yellow or emergency — no clearing
        for d in PHASE_SERVES[self._phase]:
            cleared = self._lanes[d].clear_vehicles()
            self._total_cleared += cleared

    def _handle_emergency_timer(self):
        """
        Auto-clear emergency after EMERGENCY_ALL_RED_DURATION + EMERGENCY_CLEAR_DURATION.
        In the full system the agent manages the phase transitions;
        the simulator just auto-clears the flag after a timeout.
        """
        if not self._emergency_active:
            return
        self._emergency_timer += 1
        total_duration = EMERGENCY_ALL_RED_DURATION + EMERGENCY_CLEAR_DURATION
        if self._emergency_timer >= total_duration:
            self.clear_emergency()

    def _serve_pedestrian_requests(self, phase: Phase):
        """
        When a green phase activates, clear pedestrian request flags for the
        directions whose crossings are now open (parallel to vehicle flow).
        Assumption A8: pedestrians cross in parallel with their vehicle phase.
        """
        from config import PHASE_PEDESTRIAN_CROSSINGS
        if phase in PHASE_PEDESTRIAN_CROSSINGS:
            for d in PHASE_PEDESTRIAN_CROSSINGS[phase]:
                self._pedestrian_request[d] = False

    def _maybe_generate_pedestrian_request(self):
        """
        Randomly generates pedestrian button presses each cycle.
        Simulates the stochastic nature of pedestrian arrivals.
        Each direction has a small independent probability per cycle.
        """
        for d in DIRECTIONS:
            if not self._pedestrian_request[d]:
                if random.random() < self._ped_random_prob:
                    self._pedestrian_request[d] = True

    def _advance_hour(self):
        """
        Advance simulated hour by 1 second worth of time.
        1 simulated hour = 3600 cycles. Wraps at 24.
        Also updates the traffic profile if the hour crosses a boundary.
        """
        self._hour = (self._hour + 1 / 3600) % 24
        self._update_profile_from_hour()

    def _update_profile_from_hour(self):
        """
        Automatically switch traffic profile based on simulated hour.
        Skipped if profile_locked is True (manual GUI override active).
        """
        if self.profile_locked:
            return
        from config import PROFILE_HOURS, TrafficProfile
        h = self._hour
        for profile, (start, end) in PROFILE_HOURS.items():
            if start < end:
                if start <= h < end:
                    if self._profile != profile:
                        self.set_profile(profile)
                    return
            else:
                if h >= start or h < end:
                    if self._profile != profile:
                        self.set_profile(profile)
                    return

    # =========================================================================
    # Debug / testing helpers
    # =========================================================================

    def set_queue(self, direction: int, count: int):
        """Force-set a queue length — used in unit testing."""
        self._lanes[direction]._queue = min(count, MAX_QUEUE_LENGTH)

    def get_queue(self, direction: int) -> int:
        return self._lanes[direction].count

    def get_all_queues(self) -> List[int]:
        return [self._lanes[d].count for d in DIRECTIONS]

    def summary(self) -> str:
        queues = self.get_all_queues()
        dirs   = ["N", "S", "E", "W"]
        q_str  = " | ".join(f"{dirs[i]}:{queues[i]:>2}" for i in DIRECTIONS)
        return (
            f"Cycle {self._cycle:>5} | {self._phase.name:<12} | "
            f"Queues [{q_str}] | "
            f"Cleared: {self._total_cleared:>4} | "
            f"Arrivals: {self._total_arrivals:>4} | "
            f"Hour: {self._hour:.2f} | "
            f"Profile: {self._profile.value}"
        )