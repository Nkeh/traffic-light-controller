# =============================================================================
# agent.py — Autonomous Traffic Light Controller Agent
# Implements the goal-based + utility-based intelligent agent.
# Architecture: sense() -> decide() -> act() every decision cycle.
#
# Design traceability:
#   AgentState      <- Section 6.2  (internal state variables)
#   _compute_utility <- Section 8.1  (utility function formula)
#   _decide_phase   <- Section 7    (decision flowchart)
#   _run_fsm        <- Section 6.1  (FSM transitions)
#   Edge case guards <- Section 8.5 (all six edge cases)
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

from config import (
    Phase, TrafficProfile,
    DIRECTIONS, DIRECTION_NAMES, NORTH, SOUTH, EAST, WEST,
    UTILITY_COEFFICIENTS, PHASE_DURATIONS, PEDESTRIAN_BONUS,
    YELLOW_DURATION, MAX_WAIT_BEFORE_FORCE,
    PHASE_SERVES, GREEN_TO_YELLOW, YELLOW_TO_GREEN, OPPOSING_GREEN,
    PHASE_PEDESTRIAN_CROSSINGS, PHASE_HISTORY_BUFFER,
    INITIAL_PHASE, INITIAL_PROFILE,
)
from simulator import SensorReading


# ── Decision record ───────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    """
    One entry in the agent's phase_history log.
    Stores what the agent decided, why, and what the world looked like.
    """
    cycle:          int
    phase:          Phase
    decision:       str          # human-readable reason string
    utility_ns:     float
    utility_ew:     float
    queue_snapshot: tuple        # (N, S, E, W) at time of decision
    wait_snapshot:  tuple        # (N, S, E, W) at time of decision


# ── Agent internal state ──────────────────────────────────────────────────────

@dataclass
class AgentState:
    """
    The agent's complete internal model of the world.
    Updated every cycle by sense(). Read by decide(). Written by act().

    Maps directly to Section 6.2 — Internal State Variables.
    """

    # ── Category 1: Queue & vehicle data ─────────────────────────────────────
    queue_length:   List[int]   = field(default_factory=lambda: [0, 0, 0, 0])
    wait_time:      List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    arrival_rate:   List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # ── Category 2: Current signal state ─────────────────────────────────────
    current_phase:  Phase       = INITIAL_PHASE
    phase_elapsed:  int         = 0
    min_green:      int         = 10
    max_green:      int         = 60
    yellow_duration: int        = YELLOW_DURATION   # always 3 — never changes

    # ── Category 3: Temporal context ─────────────────────────────────────────
    hour_of_day:    float       = 9.0
    traffic_profile: TrafficProfile = INITIAL_PROFILE
    cycle_count:    int         = 0

    # ── Category 4: Override flags ────────────────────────────────────────────
    emergency_active:    bool          = False
    emergency_direction: Optional[int] = None
    pedestrian_request:  List[bool]    = field(default_factory=lambda: [False]*4)

    # ── Category 5: Performance metrics ──────────────────────────────────────
    avg_wait_time:    float = 0.0
    vehicles_cleared: int   = 0
    phase_history:    deque = field(default_factory=lambda: deque(maxlen=PHASE_HISTORY_BUFFER))

    # ── Working memory (not part of original design — internal bookkeeping) ───
    last_utility_ns:   float = 0.0
    last_utility_ew:   float = 0.0
    last_decision:     str   = "initialising"
    # Emergency transition state:
    # False = no emergency pending
    # True  = emergency detected, yellow clearance in progress before all-red
    emergency_pending: bool  = False


# ── Agent ─────────────────────────────────────────────────────────────────────

class TrafficLightAgent:
    """
    Goal-based + utility-based intelligent agent for traffic signal control.

    Per-cycle operation:
        reading = simulator.get_sensor_reading()
        phase   = agent.run_cycle(reading)
        simulator.apply_phase(phase)

    The agent's job is to return the Phase that should be active
    this cycle. The simulator applies it to the world.
    """

    def __init__(
        self,
        initial_phase:   Phase          = INITIAL_PHASE,
        initial_profile: TrafficProfile = INITIAL_PROFILE,
    ):
        self.state = AgentState(
            current_phase   = initial_phase,
            traffic_profile = initial_profile,
        )
        self._sync_profile_constants(initial_profile)

    # =========================================================================
    # Public interface — called by main loop every cycle
    # =========================================================================

    def run_cycle(self, reading: SensorReading) -> Phase:
        """
        Execute one full sense -> decide -> act cycle.
        Returns the Phase the agent has decided should be active.

        This is the only method the main loop needs to call.
        """
        self._sense(reading)
        new_phase = self._decide()
        self._act(new_phase)
        return self.state.current_phase

    # ── Accessors for GUI and logger ──────────────────────────────────────────

    @property
    def state_snapshot(self) -> AgentState:
        return self.state

    @property
    def utility_scores(self) -> Tuple[float, float]:
        """Returns (utility_NS, utility_EW) from the last cycle."""
        return self.state.last_utility_ns, self.state.last_utility_ew

    @property
    def last_decision(self) -> str:
        return self.state.last_decision

    # =========================================================================
    # SENSE — update internal state from sensor reading
    # =========================================================================

    def _sense(self, reading: SensorReading):
        """
        Ingest sensor reading and update every internal state variable.
        Also updates wait_time — lanes not being served accumulate wait.
        """
        s = self.state

        s.cycle_count    = reading.cycle
        s.phase_elapsed  = reading.phase_elapsed
        s.hour_of_day    = reading.hour_of_day
        s.emergency_active    = reading.emergency_active
        s.emergency_direction = reading.emergency_direction

        # Update queue lengths from sensors
        for d in DIRECTIONS:
            s.queue_length[d]  = reading.queue_length[d]
            s.arrival_rate[d]  = reading.arrival_rate[d]
            s.pedestrian_request[d] = reading.pedestrian_request[d]

        # Update wait times:
        # Lanes being served reset to 0; others accumulate +1 each cycle.
        served = PHASE_SERVES.get(s.current_phase, [])
        for d in DIRECTIONS:
            if d in served:
                s.wait_time[d] = 0.0
            else:
                s.wait_time[d] += 1.0

        # Sync profile and its constants if changed
        if reading.traffic_profile != s.traffic_profile:
            s.traffic_profile = reading.traffic_profile
            self._sync_profile_constants(reading.traffic_profile)

    # =========================================================================
    # DECIDE — the agent's full decision logic
    # Maps directly to Section 7 (decision flowchart) and Section 8.5 (edge cases)
    # =========================================================================

    def _decide(self) -> Phase:
        """
        Run the complete decision flowchart.
        Returns the Phase that should be active after this cycle.

        Order (matches flowchart exactly):
          1. Emergency override check           — edge case 4
          2. Yellow phase passthrough           — FSM rule
          3. Minimum green lock                 — constraint C2
          4. All-queues-empty check             — edge case 2
          5. Starvation backstop                — edge case 3
          6. Compute utility for both phases    — Section 8.1
          7. Maximum green force-switch         — constraint C3
          8. Utility-based switch decision      — core logic
        """
        s = self.state

        # ── Step 1: Emergency override ─────────────────────────────────────
        # Constraint C4: bypasses ALL other logic. Fires immediately.
        if s.emergency_active:
            return self._handle_emergency()

        # ── Step 2: Yellow phase passthrough ───────────────────────────────
        # During yellow, the agent does nothing — it just waits for the
        # fixed 3-second clearance to elapse, then transitions to next green.
        if s.current_phase in (Phase.NS_YELLOW, Phase.EW_YELLOW):
            return self._run_yellow()

        # ── Step 2b: EMERGENCY phase hold ──────────────────────────────────
        # If we are in EMERGENCY but emergency_active just cleared this cycle,
        # _handle_emergency() above will resume the correct green phase.
        # If somehow we reach here still in EMERGENCY, hold it safely.
        if s.current_phase == Phase.EMERGENCY:
            return self._handle_emergency()

        # ── Step 3: Minimum green lock ─────────────────────────────────────
        # Constraint C2: no switch allowed before min_green elapsed.
        # Prevents dangerous flickering regardless of utility scores.
        if s.phase_elapsed < s.min_green:
            self._record_decision("held — below min_green")
            return s.current_phase

        # ── Step 4: All queues empty ────────────────────────────────────────
        # Edge case 2: collapse to min-cycle mode when intersection is clear.
        if self._all_queues_empty():
            return self._handle_empty_intersection()

        # ── Step 5: Starvation backstop ─────────────────────────────────────
        # Edge case 3: if any opposing lane has waited too long, force switch.
        forced = self._check_starvation()
        if forced is not None:
            return forced

        # ── Step 6: Compute utility for both green phases ──────────────────
        u_ns = self._compute_utility(Phase.NS_GREEN)
        u_ew = self._compute_utility(Phase.EW_GREEN)
        s.last_utility_ns = u_ns
        s.last_utility_ew = u_ew

        # ── Step 7: Maximum green force-switch ──────────────────────────────
        # Constraint C3: force transition regardless of utility scores.
        if s.phase_elapsed >= s.max_green:
            return self._force_switch("max_green exceeded")

        # ── Step 8: Utility-based switch decision ───────────────────────────
        return self._evaluate_switch(u_ns, u_ew)

    # =========================================================================
    # UTILITY FUNCTION — Section 8.1
    # U(phase) = alpha * queue_score + beta * wait_score - gamma * switch_penalty
    # =========================================================================

    def _compute_utility(self, candidate: Phase) -> float:
        """
        Compute the utility score for a candidate green phase.

        Formula: U = alpha * queue_score
                   + beta  * wait_score       (wait^2 for non-linear urgency)
                   - gamma * switch_penalty   (0 if staying, gamma if switching)
                   + delta * pedestrian_bonus (per active ped request served)

        Args:
            candidate: the Phase to evaluate (must be NS_GREEN or EW_GREEN)
        Returns:
            float utility score — higher is better
        """
        s    = self.state
        dirs = PHASE_SERVES[candidate]
        coef = UTILITY_COEFFICIENTS[s.traffic_profile]

        alpha = coef["alpha"]
        beta  = coef["beta"]
        gamma = coef["gamma"]
        delta = PEDESTRIAN_BONUS

        # Queue pressure term — total vehicles waiting in served lanes
        queue_score = sum(s.queue_length[d] for d in dirs)

        # Starvation prevention term — squared wait time grows non-linearly
        # A lane at 30s scores far more than one at 10s (900 vs 100)
        wait_score = sum(s.wait_time[d] ** 2 for d in dirs)

        # Switching cost — flat penalty subtracted if phase would change
        is_switch     = (candidate != s.current_phase)
        switch_penalty = 1 if is_switch else 0

        # Pedestrian bonus — nudge toward phases with pending ped requests
        ped_bonus = sum(
            delta for d in PHASE_PEDESTRIAN_CROSSINGS.get(candidate, [])
            if s.pedestrian_request[d]
        )

        utility = (alpha * queue_score
                 + beta  * wait_score
                 - gamma * switch_penalty
                 + ped_bonus)

        return round(utility, 2)

    # =========================================================================
    # FSM TRANSITIONS — Section 6.1
    # =========================================================================

    def _run_yellow(self) -> Phase:
        """
        Yellow phase: wait for fixed clearance duration then transition.
        Constraint C1: yellow_duration is always exactly 3 seconds.
        The utility function is NOT evaluated during yellow.

        If emergency_pending is True, hand off to _handle_emergency
        once yellow completes — do NOT go to the normal next green.
        """
        s = self.state
        if s.phase_elapsed >= s.yellow_duration:
            # Emergency pending — hand off to emergency handler instead of
            # transitioning to the normal next green phase
            if s.emergency_pending:
                return self._handle_emergency()
            next_phase = YELLOW_TO_GREEN[s.current_phase]
            self._record_decision(
                f"yellow complete — transitioning to {next_phase.name}"
            )
            return next_phase
        self._record_decision(
            f"yellow — {s.phase_elapsed}/{s.yellow_duration}s clearance"
        )
        return s.current_phase

    def _force_switch(self, reason: str) -> Phase:
        """Force a transition to yellow regardless of utility scores."""
        s          = self.state
        next_phase = GREEN_TO_YELLOW[s.current_phase]
        self._record_decision(f"FORCE switch to {next_phase.name} — {reason}")
        return next_phase

    def _evaluate_switch(self, u_ns: float, u_ew: float) -> Phase:
        """
        Core utility comparison.
        Switch only if the opposing phase's utility exceeds the current
        phase's utility by more than gamma (the hysteresis margin).

        Edge case 1 (tie): stays in current phase — a tie is not enough
        reason to pay the switching cost.
        """
        s     = self.state
        coef  = UTILITY_COEFFICIENTS[s.traffic_profile]
        gamma = coef["gamma"]

        is_ns = s.current_phase == Phase.NS_GREEN
        u_current  = u_ns if is_ns else u_ew
        u_opposing = u_ew if is_ns else u_ns
        opp_phase  = Phase.EW_GREEN if is_ns else Phase.NS_GREEN

        if u_opposing > u_current + gamma:
            # Switch — opposing phase has meaningfully higher utility
            next_phase = GREEN_TO_YELLOW[s.current_phase]
            self._record_decision(
                f"switching -> {next_phase.name} "
                f"(U_opp={u_opposing:.1f} > U_cur={u_current:.1f} + γ={gamma})"
            )
            return next_phase
        else:
            # Hold — current phase wins or tie
            self._record_decision(
                f"held — U({s.current_phase.name[:2]}){u_current:.1f} "
                f"U({'EW' if is_ns else 'NS'}){u_opposing:.1f} γ={gamma}"
            )
            return s.current_phase

    # =========================================================================
    # EDGE CASE HANDLERS — Section 8.5
    # =========================================================================

    def _handle_emergency(self) -> Phase:
        """
        Edge case 4 — Emergency override.
        Three-stage sequence (corrected design):

          Stage 1 — Yellow clearance (NEW):
            Emergency detected while in a green phase.
            Agent enters the corresponding yellow phase first (3s) to give
            vehicles already in the intersection time to clear safely.
            Constraint C1 respected: yellow is never skipped.

          Stage 2 — All-red (EMERGENCY):
            After yellow completes, all signals go red.
            Emergency vehicle passes through the cleared intersection.
            Held until the simulator clears emergency_active.

          Stage 3 — Resume:
            Emergency flag cleared by simulator.
            Agent resumes the green phase that serves the emergency direction.

        State machine:
          GREEN -> YELLOW (emergency_pending=True) -> EMERGENCY -> GREEN
        """
        s = self.state

        # ── Stage 3: emergency_active just cleared — resume normal ────────────
        if s.current_phase == Phase.EMERGENCY and not s.emergency_active:
            s.emergency_pending = False
            resume = self._phase_for_direction(s.emergency_direction)
            self._record_decision(
                f"emergency cleared — resuming {resume.name}"
            )
            return resume

        # ── Stage 2: holding EMERGENCY all-red ───────────────────────────────
        if s.current_phase == Phase.EMERGENCY:
            self._record_decision("EMERGENCY — all red, intersection clear")
            return Phase.EMERGENCY

        # ── Stage 1a: yellow clearance in progress ────────────────────────────
        # We are in a yellow phase with emergency_pending=True.
        # Wait for yellow to complete (handled by _run_yellow),
        # then transition to EMERGENCY instead of the normal next green.
        if s.emergency_pending and s.current_phase in (Phase.NS_YELLOW, Phase.EW_YELLOW):
            if s.phase_elapsed >= s.yellow_duration:
                # Yellow complete — now enter EMERGENCY
                self._record_decision(
                    f"yellow clearance done — entering EMERGENCY all-red "
                    f"(from {DIRECTION_NAMES.get(s.emergency_direction, '?')})"
                )
                return Phase.EMERGENCY
            else:
                self._record_decision(
                    f"EMERGENCY pending — yellow clearance "
                    f"{s.phase_elapsed}/{s.yellow_duration}s"
                )
                return s.current_phase

        # ── Stage 1b: new emergency detected — enter yellow first ─────────────
        # If already in yellow (edge case: emergency fires during yellow),
        # just set the pending flag and let yellow finish naturally.
        if s.current_phase in (Phase.NS_YELLOW, Phase.EW_YELLOW):
            s.emergency_pending = True
            self._record_decision(
                f"EMERGENCY detected mid-yellow — completing clearance first"
            )
            return s.current_phase

        # Currently in a green phase — enter yellow clearance before all-red
        s.emergency_pending = True
        yellow_phase = GREEN_TO_YELLOW[s.current_phase]
        self._record_decision(
            f"EMERGENCY from {DIRECTION_NAMES.get(s.emergency_direction, '?')} "
            f"— yellow clearance before all-red"
        )
        return yellow_phase

    def _handle_empty_intersection(self) -> Phase:
        """
        Edge case 2 — All queues empty.
        Stay in current green phase but treat min_green as the effective
        max. When min_green expires, switch to keep cycling.
        This prevents the agent from stalling on an empty road.
        """
        s = self.state
        if s.phase_elapsed >= s.min_green:
            self._record_decision("empty intersection — cycling to next phase")
            return GREEN_TO_YELLOW[s.current_phase]
        self._record_decision("empty intersection — min cycle holding")
        return s.current_phase

    def _check_starvation(self) -> Optional[Phase]:
        """
        Edge case 3 — Lane starvation backstop.
        If any lane in the opposing phase has exceeded MAX_WAIT_BEFORE_FORCE,
        force a switch immediately regardless of utility scores.
        The squared wait_score term should prevent reaching this point
        in normal operation — this is the hard safety net.
        """
        s = self.state
        # Only valid during a green phase — not during yellow or emergency
        if s.current_phase not in OPPOSING_GREEN:
            return None
        opp_phase    = OPPOSING_GREEN[s.current_phase]
        opp_dirs     = PHASE_SERVES[opp_phase]

        for d in opp_dirs:
            if s.wait_time[d] >= MAX_WAIT_BEFORE_FORCE:
                next_phase = GREEN_TO_YELLOW[s.current_phase]
                self._record_decision(
                    f"STARVATION backstop — {DIRECTION_NAMES[d]} "
                    f"waited {s.wait_time[d]:.0f}s — forcing {next_phase.name}"
                )
                return next_phase
        return None

    def _all_queues_empty(self) -> bool:
        """
        Edge case 2 check.
        Returns True only if all four queues are zero AND no pedestrian
        requests are pending. If a pedestrian is waiting, we still want
        the normal utility logic to run so their bonus is evaluated.
        """
        s = self.state
        all_zero  = all(q == 0 for q in s.queue_length)
        no_peds   = not any(s.pedestrian_request)
        return all_zero and no_peds

    # =========================================================================
    # ACT — apply decision, update performance metrics
    # =========================================================================

    def _act(self, new_phase: Phase):
        """
        Apply the decided phase to internal state.
        Updates performance metrics and logs the decision.
        """
        s = self.state

        # Update phase and reset elapsed timer on transition
        if new_phase != s.current_phase:
            s.current_phase = new_phase
            s.phase_elapsed = 0
        else:
            # phase_elapsed is already synced from sensor reading in _sense()
            pass

        # Update rolling average wait time
        total_wait   = sum(s.wait_time) / 4
        s.avg_wait_time = s.avg_wait_time * 0.95 + total_wait * 0.05

        # Append to phase history log
        s.phase_history.append(DecisionRecord(
            cycle          = s.cycle_count,
            phase          = s.current_phase,
            decision       = s.last_decision,
            utility_ns     = s.last_utility_ns,
            utility_ew     = s.last_utility_ew,
            queue_snapshot = tuple(s.queue_length),
            wait_snapshot  = tuple(s.wait_time),
        ))

    # =========================================================================
    # Helpers
    # =========================================================================

    def _sync_profile_constants(self, profile: TrafficProfile):
        """Load min_green and max_green for the given profile."""
        d = PHASE_DURATIONS[profile]
        self.state.min_green = d["min_green"]
        self.state.max_green = d["max_green"]

    def _phase_for_direction(self, direction: Optional[int]) -> Phase:
        """
        Return the green phase that serves the given direction.
        Used when resuming after an emergency override.
        """
        if direction in (NORTH, SOUTH):
            return Phase.NS_GREEN
        if direction in (EAST, WEST):
            return Phase.EW_GREEN
        return Phase.NS_GREEN   # default fallback

    def _record_decision(self, reason: str):
        """Store the human-readable decision reason for logging and GUI."""
        self.state.last_decision = reason

    # =========================================================================
    # Debug helpers
    # =========================================================================

    def status(self) -> str:
        """One-line status string — used in headless console mode."""
        s    = self.state
        u_ns, u_ew = self.utility_scores
        dirs = ["N","S","E","W"]
        q    = " ".join(f"{dirs[i]}:{s.queue_length[i]}" for i in DIRECTIONS)
        w    = " ".join(f"{dirs[i]}:{s.wait_time[i]:.0f}" for i in DIRECTIONS)
        return (
            f"[{s.cycle_count:>5}] {s.current_phase.name:<12} "
            f"el={s.phase_elapsed:>3}s | "
            f"Q[{q}] W[{w}] | "
            f"U(NS)={u_ns:>7.1f} U(EW)={u_ew:>7.1f} | "
            f"{s.last_decision}"
        )

    def utility_breakdown(self, candidate: Phase) -> str:
        """
        Detailed breakdown of the utility calculation for a given phase.
        Used for debugging and academic demonstration.
        """
        s    = self.state
        dirs = PHASE_SERVES[candidate]
        coef = UTILITY_COEFFICIENTS[s.traffic_profile]

        alpha = coef["alpha"]
        beta  = coef["beta"]
        gamma = coef["gamma"]

        q_score = sum(s.queue_length[d] for d in dirs)
        w_score = sum(s.wait_time[d] ** 2 for d in dirs)
        is_sw   = candidate != s.current_phase
        sw_pen  = gamma if is_sw else 0
        ped_b   = sum(
            PEDESTRIAN_BONUS for d in PHASE_PEDESTRIAN_CROSSINGS.get(candidate,[])
            if s.pedestrian_request[d]
        )
        total = alpha*q_score + beta*w_score - sw_pen + ped_b

        label = candidate.name
        return (
            f"U({label}) = "
            f"{alpha}×{q_score}(q) + {beta}×{w_score:.0f}(w²) "
            f"- {sw_pen:.1f}(switch) + {ped_b:.1f}(ped) "
            f"= {total:.2f}"
        )