# =============================================================================
# config.py — Autonomous Traffic Light Controller
# All constants, coefficients, thresholds, and configuration values.
# Every other module imports from here. Change values here only.
# =============================================================================

from enum import Enum, auto


# ── Phase definitions ─────────────────────────────────────────────────────────

class Phase(Enum):
    NS_GREEN  = auto()   # North + South green,  East + West red
    NS_YELLOW = auto()   # North + South yellow, East + West red
    EW_GREEN  = auto()   # East  + West  green,  North + South red
    EW_YELLOW = auto()   # East  + West  yellow, North + South red
    EMERGENCY = auto()   # All directions red — emergency override


# ── Direction indices ─────────────────────────────────────────────────────────
# Used everywhere as list indices: queues[NORTH], waits[EAST], etc.

NORTH = 0
SOUTH = 1
EAST  = 2
WEST  = 3
DIRECTIONS = [NORTH, SOUTH, EAST, WEST]
DIRECTION_NAMES = {NORTH: "North", SOUTH: "South", EAST: "East", WEST: "West"}


# ── Traffic profile definitions ───────────────────────────────────────────────

class TrafficProfile(Enum):
    MORNING_RUSH = "MORNING_RUSH"
    MIDDAY       = "MIDDAY"
    EVENING_RUSH = "EVENING_RUSH"
    NIGHT        = "NIGHT"


# Hour ranges that map to each profile (24-hour clock)
PROFILE_HOURS = {
    TrafficProfile.MORNING_RUSH: (7,  9),
    TrafficProfile.MIDDAY:       (9,  16),
    TrafficProfile.EVENING_RUSH: (16, 19),
    TrafficProfile.NIGHT:        (19, 7),   # wraps midnight
}


# ── Utility function coefficients ─────────────────────────────────────────────
# Per profile: alpha (queue weight), beta (wait weight), gamma (switch penalty)
# delta is the pedestrian bonus — same across all profiles

UTILITY_COEFFICIENTS = {
    #                               alpha  beta  gamma
    TrafficProfile.MORNING_RUSH: {  "alpha": 2.0, "beta": 1.5, "gamma": 5.0 },
    TrafficProfile.MIDDAY:       {  "alpha": 1.5, "beta": 1.5, "gamma": 3.0 },
    TrafficProfile.EVENING_RUSH: {  "alpha": 2.0, "beta": 2.0, "gamma": 5.0 },
    TrafficProfile.NIGHT:        {  "alpha": 1.0, "beta": 2.5, "gamma": 2.0 },
}

PEDESTRIAN_BONUS = 8.0   # delta — added to utility of phase serving a ped request


# ── Phase duration constraints (seconds) ─────────────────────────────────────
# Hard limits — the FSM enforces these before the utility function is evaluated.

YELLOW_DURATION = 3       # Fixed clearance time between any two green phases (safety-critical)

# Min/max green per profile (in seconds)
PHASE_DURATIONS = {
    #                               min_green  max_green
    TrafficProfile.MORNING_RUSH: { "min_green": 12, "max_green": 90 },
    TrafficProfile.MIDDAY:       { "min_green": 10, "max_green": 60 },
    TrafficProfile.EVENING_RUSH: { "min_green": 12, "max_green": 90 },
    TrafficProfile.NIGHT:        { "min_green":  8, "max_green": 40 },
}


# ── Starvation backstop ───────────────────────────────────────────────────────
# If any lane's wait_time exceeds this, green is forced regardless of utility.
# This is the hard backstop for edge case 3 (lane starvation).

MAX_WAIT_BEFORE_FORCE = 120   # seconds


# ── Phase groupings ───────────────────────────────────────────────────────────
# Which directions each green phase serves, and which yellow follows each green.

PHASE_SERVES = {
    Phase.NS_GREEN:  [NORTH, SOUTH],
    Phase.EW_GREEN:  [EAST,  WEST],
}

PHASE_PEDESTRIAN_CROSSINGS = {
    Phase.NS_GREEN: [NORTH, SOUTH],   # N+S vehicles flow → N+S pedestrians cross
    Phase.EW_GREEN: [EAST,  WEST],    # E+W vehicles flow → E+W pedestrians cross
}

GREEN_TO_YELLOW = {
    Phase.NS_GREEN:  Phase.NS_YELLOW,
    Phase.EW_GREEN:  Phase.EW_YELLOW,
}

YELLOW_TO_GREEN = {
    Phase.NS_YELLOW: Phase.EW_GREEN,
    Phase.EW_YELLOW: Phase.NS_GREEN,
}

OPPOSING_GREEN = {
    Phase.NS_GREEN: Phase.EW_GREEN,
    Phase.EW_GREEN: Phase.NS_GREEN,
}


# ── Simulation parameters ─────────────────────────────────────────────────────

CYCLE_DURATION_MS    = 1000    # milliseconds per decision cycle (1 second real-time)
SIMULATION_SPEED     = 1.0     # multiplier — 1.0 = real time, 2.0 = double speed
INITIAL_PHASE        = Phase.NS_GREEN
INITIAL_PROFILE      = TrafficProfile.MIDDAY
RANDOM_SEED          = 42      # set to None for non-deterministic runs


# ── Vehicle arrival rates (vehicles per second per lane) ──────────────────────
# Used by the simulator's stochastic arrival model (Poisson process).
# [NORTH, SOUTH, EAST, WEST]

ARRIVAL_RATES = {
    TrafficProfile.MORNING_RUSH: [0.40, 0.35, 0.20, 0.20],
    TrafficProfile.MIDDAY:       [0.20, 0.18, 0.15, 0.15],
    TrafficProfile.EVENING_RUSH: [0.25, 0.28, 0.38, 0.35],
    TrafficProfile.NIGHT:        [0.05, 0.05, 0.05, 0.05],
}

# Max vehicles that can clear per green cycle per lane (saturation flow rate)
SATURATION_FLOW_RATE = 2      # vehicles cleared per second of green

# Maximum queue length the simulator will allow (road capacity)
MAX_QUEUE_LENGTH = 25


# ── Emergency vehicle settings ────────────────────────────────────────────────
# Sequence: GREEN -> YELLOW (3s clearance) -> EMERGENCY all-red -> resume GREEN

EMERGENCY_ALL_RED_DURATION = 5    # seconds all-red held while emergency vehicle passes
EMERGENCY_CLEAR_DURATION   = 8    # seconds green held for emergency direction after all-red


# ── Logging & output ──────────────────────────────────────────────────────────

LOG_FILE             = "simulation_log.csv"
LOG_HEADERS          = [
    "cycle", "timestamp_s", "profile",
    "phase",
    "queue_N", "queue_S", "queue_E", "queue_W",
    "wait_N",  "wait_S",  "wait_E",  "wait_W",
    "utility_NS", "utility_EW",
    "decision",
    "vehicles_cleared", "avg_wait_time",
    "emergency_active", "ped_request_N", "ped_request_S",
    "ped_request_E", "ped_request_W",
]
PHASE_HISTORY_BUFFER = 50         # number of past phase transitions to keep in memory


# ── GUI settings (Pygame) ─────────────────────────────────────────────────────

WINDOW_TITLE         = "Autonomous Traffic Light Controller"
WINDOW_WIDTH         = 1100
WINDOW_HEIGHT        = 700
TARGET_FPS           = 60

# Colours (RGB)
COLOUR_BG            = (18,  18,  42)
COLOUR_ROAD          = (45,  45,  74)
COLOUR_CENTRE        = (37,  37,  64)
COLOUR_PANEL_BG      = (18,  18,  40)
COLOUR_PANEL_BORDER  = (42,  42,  74)
COLOUR_RED_ON        = (255, 59,  59)
COLOUR_AMBER_ON      = (255, 170, 0)
COLOUR_GREEN_ON      = (0,   230, 118)
COLOUR_BULB_OFF      = (34,  34,  64)
COLOUR_QUEUE_NS      = (91,  106, 255)
COLOUR_QUEUE_EW      = (155, 106, 255)
COLOUR_CROSSING_WALK = (0,   230, 118)
COLOUR_CROSSING_WAIT = (255, 59,  59)
COLOUR_TEXT_PRIMARY  = (224, 224, 255)
COLOUR_TEXT_MUTED    = (136, 136, 204)
COLOUR_TEXT_DIM      = (90,  90,  138)
COLOUR_EMERGENCY     = (255, 68,  68)

# Fonts (loaded at runtime by gui.py)
FONT_MONO_SIZES      = [10, 11, 12, 14, 16, 20, 24]


# ── Sanity check (runs on import) ─────────────────────────────────────────────

def _validate():
    assert YELLOW_DURATION > 0,          "YELLOW_DURATION must be positive"
    for profile in TrafficProfile:
        d = PHASE_DURATIONS[profile]
        assert d["min_green"] < d["max_green"], \
            f"{profile}: min_green must be less than max_green"
        c = UTILITY_COEFFICIENTS[profile]
        assert c["alpha"] > 0 and c["beta"] > 0 and c["gamma"] > 0, \
            f"{profile}: all coefficients must be positive"
        rates = ARRIVAL_RATES[profile]
        assert len(rates) == 4, \
            f"{profile}: arrival rates must have exactly 4 values"
    assert SATURATION_FLOW_RATE > 0,     "SATURATION_FLOW_RATE must be positive"
    assert MAX_QUEUE_LENGTH > 0,         "MAX_QUEUE_LENGTH must be positive"
    assert WINDOW_WIDTH > 0 and WINDOW_HEIGHT > 0, "Window dimensions must be positive"

_validate()