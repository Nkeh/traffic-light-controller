# =============================================================================
# gui.py — Pygame Visual Simulation
# Renders the intersection, signal lights, queues, pedestrian crossings,
# and the live stats panel. Runs the main simulation loop at TARGET_FPS.
#
# Layout (1100 x 700):
#   Left  (740px) — intersection canvas
#   Right (360px) — stats panel
#   Bottom bar    — controls
# =============================================================================

import sys
import time
import pygame

from config import (
    Phase, TrafficProfile,
    NORTH, SOUTH, EAST, WEST, DIRECTIONS, DIRECTION_NAMES,
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, TARGET_FPS,
    CYCLE_DURATION_MS,
    COLOUR_BG, COLOUR_ROAD, COLOUR_CENTRE, COLOUR_PANEL_BG,
    COLOUR_PANEL_BORDER, COLOUR_RED_ON, COLOUR_AMBER_ON, COLOUR_GREEN_ON,
    COLOUR_BULB_OFF, COLOUR_QUEUE_NS, COLOUR_QUEUE_EW,
    COLOUR_CROSSING_WALK, COLOUR_CROSSING_WAIT,
    COLOUR_TEXT_PRIMARY, COLOUR_TEXT_MUTED, COLOUR_TEXT_DIM,
    COLOUR_EMERGENCY,
)
from simulator import IntersectionSimulator
from agent     import TrafficLightAgent
from logger    import SimulationLogger
from main      import run_one_cycle


# ── Colour helpers ────────────────────────────────────────────────────────────

def lerp_colour(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def with_alpha(surf, colour, alpha, rect):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    s.fill((*colour, alpha))
    surf.blit(s, (rect[0], rect[1]))


# ── Layout constants ──────────────────────────────────────────────────────────

CANVAS_W    = 740
PANEL_W     = WINDOW_WIDTH - CANVAS_W       # 360
CTRL_H      = 52
CANVAS_H    = WINDOW_HEIGHT - CTRL_H        # 648

# Intersection geometry (centred in canvas)
IX = CANVAS_W  // 2        # intersection centre x
IY = CANVAS_H  // 2 - 10  # intersection centre y
ROAD_W      = 90           # road width (pixels)
LANE_W      = ROAD_W // 2  # single lane width

# Signal box dimensions
SIG_W, SIG_H   = 22, 58
BULB_R         = 7

# Queue bar dimensions
Q_MAX_LEN      = 120       # max bar length in pixels (= MAX_QUEUE_LENGTH veh)
Q_THICK        = 18        # bar thickness

# Crossing stripe geometry
STRIPE_COUNT   = 7
STRIPE_W       = 8
STRIPE_GAP     = 4

# Panel layout
PX             = CANVAS_W + 12    # panel content left edge
PY_START       = 14               # panel top padding

# Control bar
CY             = WINDOW_HEIGHT - CTRL_H


# ── Font sizes ────────────────────────────────────────────────────────────────

F_TINY   = 11
F_SMALL  = 13
F_MED    = 15
F_LARGE  = 18
F_TITLE  = 20


# =============================================================================
# TrafficGUI
# =============================================================================

class TrafficGUI:
    """
    Pygame-based visual simulation of the traffic light controller.
    Renders every element of the mockup in native Pygame drawing calls.
    """

    def __init__(
        self,
        simulator: IntersectionSimulator,
        agent:     TrafficLightAgent,
        logger:    SimulationLogger,
    ):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        self.screen  = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock   = pygame.time.Clock()

        self.sim     = simulator
        self.agent   = agent
        self.logger  = logger

        # Fonts (monospace for data, sans for labels)
        self._load_fonts()

        # Simulation state
        self.running      = True
        self.paused       = False
        self.speed        = 2          # cycles per second
        self._cycle_accum = 0.0       # accumulated time for pacing
        self._last_tick   = time.time()

        # Animation state
        self._phase_flash    = 0.0    # 0-1 pulse for current phase badge
        self._flash_dir      = 1
        self._queue_anim     = [0.0] * 4   # smooth bar lengths
        self._ped_flash      = [0.0] * 4   # pedestrian crossing pulse

        # Button rects (built during first draw)
        self._btn_pause    = pygame.Rect(0, 0, 0, 0)
        self._btn_emrg     = pygame.Rect(0, 0, 0, 0)
        self._btn_ped_n    = pygame.Rect(0, 0, 0, 0)
        self._btn_ped_e    = pygame.Rect(0, 0, 0, 0)
        self._btn_ped_s    = pygame.Rect(0, 0, 0, 0)
        self._btn_ped_w    = pygame.Rect(0, 0, 0, 0)
        self._profile_btns: list = []
        self._speed_btns:   list = []

        # Log lines ring buffer (shown in decision log panel)
        self._log_lines: list = []
        self._prev_decision = ""

    # =========================================================================
    # Main loop
    # =========================================================================

    def run(self):
        """Main event + render loop."""
        while self.running:
            dt = self.clock.tick(TARGET_FPS) / 1000.0   # seconds since last frame

            self._handle_events()
            self._update(dt)
            self._draw()
            pygame.display.flip()

        pygame.quit()

    # =========================================================================
    # Event handling
    # =========================================================================

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_e:
            self._trigger_emergency()
        elif key == pygame.K_1:
            self.speed = 1
        elif key == pygame.K_2:
            self.speed = 2
        elif key == pygame.K_4:
            self.speed = 4
        elif key == pygame.K_8:
            self.speed = 8

    def _handle_click(self, pos):
        mx, my = pos
        # Pause / Resume
        if self._btn_pause.collidepoint(mx, my):
            self.paused = not self.paused
        # Emergency
        elif self._btn_emrg.collidepoint(mx, my):
            self._trigger_emergency()
        # Pedestrian buttons
        elif self._btn_ped_n.collidepoint(mx, my):
            self.sim.press_pedestrian_button(NORTH)
            self._add_log("Pedestrian button: North", warn=False)
        elif self._btn_ped_s.collidepoint(mx, my):
            self.sim.press_pedestrian_button(SOUTH)
            self._add_log("Pedestrian button: South", warn=False)
        elif self._btn_ped_e.collidepoint(mx, my):
            self.sim.press_pedestrian_button(EAST)
            self._add_log("Pedestrian button: East", warn=False)
        elif self._btn_ped_w.collidepoint(mx, my):
            self.sim.press_pedestrian_button(WEST)
            self._add_log("Pedestrian button: West", warn=False)
        # Profile buttons
        for btn_rect, profile in self._profile_btns:
            if btn_rect.collidepoint(mx, my):
                self.sim.set_profile(profile)
                self._add_log(f"Profile -> {profile.value}", warn=False)
        # Speed buttons
        for btn_rect, spd in self._speed_btns:
            if btn_rect.collidepoint(mx, my):
                self.speed = spd

    def _trigger_emergency(self):
        import random
        direction = random.choice(DIRECTIONS)
        self.sim.trigger_emergency(direction)
        self._add_log(
            f"EMERGENCY from {DIRECTION_NAMES[direction]}!", warn=True
        )

    # =========================================================================
    # Simulation update
    # =========================================================================

    def _update(self, dt: float):
        """Advance simulation and update animation state."""
        if not self.paused:
            self._cycle_accum += dt * self.speed
            while self._cycle_accum >= 1.0:
                run_one_cycle(self.sim, self.agent, self.logger)
                self._cycle_accum -= 1.0
                # Log new decisions
                dec = self.agent.last_decision
                if dec != self._prev_decision:
                    warn = any(w in dec for w in
                               ["EMERGENCY","FORCE","STARVATION","switching"])
                    self._add_log(f"[{self.sim.cycle}] {dec}", warn=warn)
                    self._prev_decision = dec

        # Animate phase badge pulse
        self._phase_flash += self._flash_dir * dt * 2
        if self._phase_flash >= 1.0:
            self._phase_flash = 1.0; self._flash_dir = -1
        elif self._phase_flash <= 0.0:
            self._phase_flash = 0.0; self._flash_dir = 1

        # Smooth queue bar lengths
        s = self.agent.state
        for d in DIRECTIONS:
            target = s.queue_length[d] / 25.0  # 0-1
            self._queue_anim[d] += (target - self._queue_anim[d]) * min(dt * 8, 1.0)

        # Pedestrian crossing flash
        for d in DIRECTIONS:
            if s.pedestrian_request[d]:
                self._ped_flash[d] = min(1.0, self._ped_flash[d] + dt * 3)
            else:
                self._ped_flash[d] = max(0.0, self._ped_flash[d] - dt * 3)

    # =========================================================================
    # Drawing — master
    # =========================================================================

    def _draw(self):
        self.screen.fill(COLOUR_BG)
        self._draw_canvas()
        self._draw_panel()
        self._draw_controls()

    # =========================================================================
    # Canvas — intersection
    # =========================================================================

    def _draw_canvas(self):
        s = self.agent.state
        phase = s.current_phase

        # ── Roads ─────────────────────────────────────────────────────────────
        road_h = pygame.Rect(0, IY - ROAD_W//2, CANVAS_W, ROAD_W)
        road_v = pygame.Rect(IX - ROAD_W//2, 0, ROAD_W, CANVAS_H)
        pygame.draw.rect(self.screen, COLOUR_ROAD, road_h)
        pygame.draw.rect(self.screen, COLOUR_ROAD, road_v)

        # Centre box
        centre = pygame.Rect(IX - ROAD_W//2, IY - ROAD_W//2, ROAD_W, ROAD_W)
        pygame.draw.rect(self.screen, COLOUR_CENTRE, centre)

        # Lane dividers (dashed centre lines)
        self._draw_dashes_h(IY, 0, IX - ROAD_W//2, 8, 5)
        self._draw_dashes_h(IY, IX + ROAD_W//2, CANVAS_W, 8, 5)
        self._draw_dashes_v(IX, 0, IY - ROAD_W//2, 8, 5)
        self._draw_dashes_v(IX, IY + ROAD_W//2, CANVAS_H, 8, 5)

        # ── Pedestrian crossings ──────────────────────────────────────────────
        self._draw_crossings(phase)

        # ── Queue bars ────────────────────────────────────────────────────────
        self._draw_queues(s)

        # ── Signal boxes ──────────────────────────────────────────────────────
        self._draw_signals(phase)

        # ── Direction labels & pedestrian buttons ─────────────────────────────
        self._draw_direction_labels()

        # ── Emergency overlay ─────────────────────────────────────────────────
        if phase == Phase.EMERGENCY:
            self._draw_emergency_overlay()

        # ── Canvas title ──────────────────────────────────────────────────────
        title = self._font(F_SMALL)
        txt = title.render("4-Way Intersection — Top View", True, COLOUR_TEXT_DIM)
        self.screen.blit(txt, (10, 8))

    def _draw_dashes_h(self, y, x0, x1, dash, gap):
        x = x0
        while x < x1:
            pygame.draw.line(self.screen, COLOUR_CENTRE,
                             (x, y), (min(x + dash, x1), y), 1)
            x += dash + gap

    def _draw_dashes_v(self, x, y0, y1, dash, gap):
        y = y0
        while y < y1:
            pygame.draw.line(self.screen, COLOUR_CENTRE,
                             (x, y), (x, min(y + dash, y1)), 1)
            y += dash + gap

    # ── Crossings ─────────────────────────────────────────────────────────────

    def _draw_crossings(self, phase):
        """Draw zebra crossing stripes — bright when active, dim when not."""
        ns_active = phase == Phase.NS_GREEN
        ew_active = phase == Phase.EW_GREEN

        # North crossing (horizontal stripes, above intersection)
        cx_n_y = IY - ROAD_W//2 - 18
        self._draw_h_crossing(IX - ROAD_W//2, IX + ROAD_W//2,
                               cx_n_y, ns_active,
                               self._ped_flash[NORTH])

        # South crossing
        cx_s_y = IY + ROAD_W//2 + 6
        self._draw_h_crossing(IX - ROAD_W//2, IX + ROAD_W//2,
                               cx_s_y, ns_active,
                               self._ped_flash[SOUTH])

        # East crossing (vertical stripes, right of intersection)
        cx_e_x = IX + ROAD_W//2 + 6
        self._draw_v_crossing(cx_e_x, IY - ROAD_W//2, IY + ROAD_W//2,
                               ew_active, self._ped_flash[EAST])

        # West crossing
        cx_w_x = IX - ROAD_W//2 - 18
        self._draw_v_crossing(cx_w_x, IY - ROAD_W//2, IY + ROAD_W//2,
                               ew_active, self._ped_flash[WEST])

    def _draw_h_crossing(self, x0, x1, y, active, ped_flash):
        colour = COLOUR_CROSSING_WALK if active else COLOUR_CROSSING_WAIT
        alpha  = 200 if active else 60
        if ped_flash > 0:
            colour = lerp_colour(colour, (255, 255, 255), ped_flash * 0.4)
            alpha  = min(255, alpha + int(ped_flash * 80))
        step = STRIPE_W + STRIPE_GAP
        x = x0
        while x < x1:
            w = min(STRIPE_W, x1 - x)
            with_alpha(self.screen, colour, alpha,
                       (x, y, w, 14))
            x += step

    def _draw_v_crossing(self, x, y0, y1, active, ped_flash):
        colour = COLOUR_CROSSING_WALK if active else COLOUR_CROSSING_WAIT
        alpha  = 200 if active else 60
        if ped_flash > 0:
            colour = lerp_colour(colour, (255, 255, 255), ped_flash * 0.4)
            alpha  = min(255, alpha + int(ped_flash * 80))
        step = STRIPE_W + STRIPE_GAP
        y = y0
        while y < y1:
            h = min(STRIPE_W, y1 - y)
            with_alpha(self.screen, colour, alpha,
                       (x, y, 14, h))
            y += step

    # ── Queue bars ────────────────────────────────────────────────────────────

    def _draw_queues(self, state):
        for d in DIRECTIONS:
            length = int(self._queue_anim[d] * Q_MAX_LEN)
            colour = COLOUR_QUEUE_NS if d in (NORTH, SOUTH) else COLOUR_QUEUE_EW
            count  = state.queue_length[d]
            if length < 2 and count == 0:
                continue

            if d == NORTH:
                rect = pygame.Rect(IX + 4, IY - ROAD_W//2 - length,
                                   Q_THICK, max(length, 2))
            elif d == SOUTH:
                rect = pygame.Rect(IX - 4 - Q_THICK, IY + ROAD_W//2,
                                   Q_THICK, max(length, 2))
            elif d == EAST:
                rect = pygame.Rect(IX + ROAD_W//2, IY + 4,
                                   max(length, 2), Q_THICK)
            else:  # WEST
                rect = pygame.Rect(IX - ROAD_W//2 - length, IY - 4 - Q_THICK,
                                   max(length, 2), Q_THICK)

            with_alpha(self.screen, colour, 160, rect)
            # Vehicle count label
            label = self._font(F_TINY).render(str(count), True, colour)
            if d == NORTH:
                self.screen.blit(label, (rect.x + 3, rect.y - 14))
            elif d == SOUTH:
                self.screen.blit(label, (rect.x + 3, rect.bottom + 2))
            elif d == EAST:
                self.screen.blit(label, (rect.right + 2, rect.y + 3))
            else:
                self.screen.blit(label, (rect.x - 14, rect.y + 3))

    # ── Signal boxes ──────────────────────────────────────────────────────────

    def _draw_signals(self, phase):
        # Determine signal colours per direction
        sig = self._phase_to_signals(phase)
        positions = {
            NORTH: (IX - ROAD_W//2 - SIG_W - 4, IY - SIG_H//2),
            SOUTH: (IX + ROAD_W//2 + 4,           IY - SIG_H//2),
            EAST:  (IX - SIG_H//2,                IY + ROAD_W//2 + 4),
            WEST:  (IX - SIG_H//2,                IY - ROAD_W//2 - SIG_W - 4),
        }
        for d in DIRECTIONS:
            state_r, state_a, state_g = sig[d]
            if d in (NORTH, SOUTH):
                x, y = positions[d]
                box  = pygame.Rect(x, y, SIG_W, SIG_H)
                self._draw_signal_box_v(box, state_r, state_a, state_g)
            else:
                x, y = positions[d]
                box  = pygame.Rect(x, y, SIG_H, SIG_W)
                self._draw_signal_box_h(box, state_r, state_a, state_g)

    def _phase_to_signals(self, phase):
        """Return (red, amber, green) booleans for each direction."""
        all_red   = (True,  False, False)
        all_amber = (False, True,  False)
        all_green = (False, False, True)
        if phase == Phase.NS_GREEN:
            return {NORTH: all_green, SOUTH: all_green,
                    EAST:  all_red,   WEST:  all_red}
        elif phase == Phase.NS_YELLOW:
            return {NORTH: all_amber, SOUTH: all_amber,
                    EAST:  all_red,   WEST:  all_red}
        elif phase == Phase.EW_GREEN:
            return {NORTH: all_red,   SOUTH: all_red,
                    EAST:  all_green, WEST:  all_green}
        elif phase == Phase.EW_YELLOW:
            return {NORTH: all_red,   SOUTH: all_red,
                    EAST:  all_amber, WEST:  all_amber}
        else:   # EMERGENCY
            return {d: all_red for d in DIRECTIONS}

    def _draw_signal_box_v(self, box, r, a, g):
        pygame.draw.rect(self.screen, (20, 20, 45), box, border_radius=4)
        cx = box.centerx
        spacing = box.height // 4
        for i, (on, col) in enumerate([(r, COLOUR_RED_ON),
                                        (a, COLOUR_AMBER_ON),
                                        (g, COLOUR_GREEN_ON)]):
            cy = box.y + spacing * (i + 1)
            pygame.draw.circle(self.screen,
                               col if on else COLOUR_BULB_OFF,
                               (cx, cy), BULB_R)

    def _draw_signal_box_h(self, box, r, a, g):
        pygame.draw.rect(self.screen, (20, 20, 45), box, border_radius=4)
        cy = box.centery
        spacing = box.width // 4
        for i, (on, col) in enumerate([(r, COLOUR_RED_ON),
                                        (a, COLOUR_AMBER_ON),
                                        (g, COLOUR_GREEN_ON)]):
            cx = box.x + spacing * (i + 1)
            pygame.draw.circle(self.screen,
                               col if on else COLOUR_BULB_OFF,
                               (cx, cy), BULB_R)

    # ── Direction labels & pedestrian buttons ──────────────────────────────────

    def _draw_direction_labels(self):
        labels = {
            NORTH: (IX, 20),
            SOUTH: (IX, CANVAS_H - 20),
            EAST:  (CANVAS_W - 20, IY),
            WEST:  (20, IY),
        }
        arrows = {NORTH: "^ N", SOUTH: "v S", EAST: "E >", WEST: "< W"}
        font   = self._font(F_SMALL)
        for d, (x, y) in labels.items():
            txt = font.render(arrows[d], True, COLOUR_TEXT_DIM)
            r   = txt.get_rect(center=(x, y))
            self.screen.blit(txt, r)

        # Pedestrian crossing buttons (clickable)
        ped_font  = self._font(F_TINY)
        ped_label = "PED"

        btn_size = 36
        btn_positions = {
            NORTH: (IX + ROAD_W//2 + 20,     IY - ROAD_W//2 - 24),
            SOUTH: (IX - ROAD_W//2 - 56,     IY + ROAD_W//2 + 6),
            EAST:  (IX + ROAD_W//2 + 6,      IY - ROAD_W//2 - 46),
            WEST:  (IX - ROAD_W//2 - 46,     IY + ROAD_W//2 + 6),
        }
        btn_map  = {NORTH: '_btn_ped_n', SOUTH: '_btn_ped_s',
                    EAST:  '_btn_ped_e', WEST:  '_btn_ped_w'}
        req      = self.agent.state.pedestrian_request

        for d, (bx, by) in btn_positions.items():
            btn_rect = pygame.Rect(bx, by, btn_size, btn_size)
            setattr(self, btn_map[d], btn_rect)
            col  = COLOUR_CROSSING_WALK if req[d] else (50, 50, 80)
            border = COLOUR_CROSSING_WALK if req[d] else (80, 80, 110)
            pygame.draw.rect(self.screen, col, btn_rect, border_radius=4)
            pygame.draw.rect(self.screen, border, btn_rect, 1, border_radius=4)
            t  = ped_font.render(ped_label, True, COLOUR_TEXT_PRIMARY)
            tr = t.get_rect(center=btn_rect.center)
            self.screen.blit(t, tr)

    # ── Emergency overlay ──────────────────────────────────────────────────────

    def _draw_emergency_overlay(self):
        pulse = abs(self._phase_flash - 0.5) * 2
        alpha = int(30 + pulse * 50)
        with_alpha(self.screen, COLOUR_EMERGENCY, alpha,
                   (0, 0, CANVAS_W, CANVAS_H))
        font = self._font(F_LARGE)
        txt  = font.render("[!] EMERGENCY OVERRIDE - ALL RED", True, COLOUR_EMERGENCY)
        r    = txt.get_rect(center=(CANVAS_W//2, 40))
        self.screen.blit(txt, r)

    # =========================================================================
    # Panel — stats
    # =========================================================================

    def _draw_panel(self):
        # Panel background
        panel_rect = pygame.Rect(CANVAS_W, 0, PANEL_W, CANVAS_H)
        pygame.draw.rect(self.screen, COLOUR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOUR_PANEL_BORDER,
                         (CANVAS_W, 0), (CANVAS_W, CANVAS_H), 1)

        s   = self.agent.state
        y   = PY_START

        y = self._panel_phase(s, y)
        y = self._panel_queues(s, y)
        y = self._panel_waits(s, y)
        y = self._panel_utility(s, y)
        y = self._panel_metrics(s, y)
        y = self._panel_log(y)

    def _panel_section(self, title: str, y: int) -> int:
        """Draw a section header and return new y."""
        font = self._font(F_TINY)
        txt  = font.render(title.upper(), True, COLOUR_TEXT_DIM)
        self.screen.blit(txt, (PX, y))
        return y + 16

    def _panel_divider(self, y: int) -> int:
        pygame.draw.line(self.screen, COLOUR_PANEL_BORDER,
                         (PX - 4, y + 4), (WINDOW_WIDTH - 8, y + 4), 1)
        return y + 12

    def _panel_phase(self, s, y: int) -> int:
        y = self._panel_section("Current Phase", y)

        # Phase badge
        phase_colours = {
            Phase.NS_GREEN:  ((0, 60, 30),    COLOUR_GREEN_ON),
            Phase.NS_YELLOW: ((50, 35, 0),    COLOUR_AMBER_ON),
            Phase.EW_GREEN:  ((20, 0, 60),    (180, 120, 255)),
            Phase.EW_YELLOW: ((50, 35, 0),    COLOUR_AMBER_ON),
            Phase.EMERGENCY: ((60, 0, 0),     COLOUR_EMERGENCY),
        }
        bg, fg = phase_colours.get(s.current_phase, ((30,30,50), COLOUR_TEXT_PRIMARY))
        badge  = pygame.Rect(PX, y, PANEL_W - 24, 28)
        pygame.draw.rect(self.screen, bg, badge, border_radius=4)
        txt = self._font(F_MED).render(s.current_phase.name, True, fg)
        r   = txt.get_rect(center=badge.center)
        self.screen.blit(txt, r)
        y  += 32

        # Phase timer bar
        is_yellow = s.current_phase in (Phase.NS_YELLOW, Phase.EW_YELLOW)
        max_t     = s.yellow_duration if is_yellow else s.max_green
        fill_pct  = min(1.0, s.phase_elapsed / max(max_t, 1))
        bar_w     = PANEL_W - 24
        bar_h     = 6
        bar_bg    = pygame.Rect(PX, y, bar_w, bar_h)
        bar_fill  = pygame.Rect(PX, y, int(bar_w * fill_pct), bar_h)
        pygame.draw.rect(self.screen, (40, 40, 70), bar_bg, border_radius=3)
        fill_col  = COLOUR_AMBER_ON if fill_pct > 0.8 else COLOUR_GREEN_ON
        if fill_pct > 0:
            pygame.draw.rect(self.screen, fill_col, bar_fill, border_radius=3)
        # Timer label
        timer_txt = self._font(F_TINY).render(
            f"{s.phase_elapsed}s / {max_t}s", True, COLOUR_TEXT_MUTED)
        self.screen.blit(timer_txt, (PX, y + 10))
        y += 26

        return self._panel_divider(y)

    def _panel_queues(self, s, y: int) -> int:
        y = self._panel_section("Queue Lengths", y)
        bar_max   = PANEL_W - 90
        dirs      = ["N","S","E","W"]
        colours   = [COLOUR_QUEUE_NS]*2 + [COLOUR_QUEUE_EW]*2
        for d in DIRECTIONS:
            q   = s.queue_length[d]
            pct = min(1.0, q / 25.0)
            # Label
            lbl = self._font(F_TINY).render(dirs[d], True, COLOUR_TEXT_MUTED)
            self.screen.blit(lbl, (PX, y + 1))
            # Bar background
            bar_bg = pygame.Rect(PX + 16, y, bar_max, 10)
            pygame.draw.rect(self.screen, (30, 30, 55), bar_bg, border_radius=3)
            # Bar fill
            if pct > 0:
                bar_fill = pygame.Rect(PX + 16, y, max(4, int(bar_max * pct)), 10)
                pygame.draw.rect(self.screen, colours[d], bar_fill, border_radius=3)
            # Count
            cnt = self._font(F_TINY).render(str(q), True, COLOUR_TEXT_MUTED)
            self.screen.blit(cnt, (PX + 16 + bar_max + 4, y + 1))
            y  += 16
        return self._panel_divider(y)

    def _panel_waits(self, s, y: int) -> int:
        y = self._panel_section("Wait Times", y)
        dirs  = ["N","S","E","W"]
        w     = (PANEL_W - 24) // 4
        for d in DIRECTIONS:
            bx  = PX + d * w
            wt  = s.wait_time[d]
            col = COLOUR_EMERGENCY if wt > 60 else (
                  COLOUR_AMBER_ON   if wt > 30 else COLOUR_GREEN_ON)
            card = pygame.Rect(bx, y, w - 4, 38)
            pygame.draw.rect(self.screen, (25, 25, 50), card, border_radius=4)
            # Direction
            dl = self._font(F_TINY).render(dirs[d], True, COLOUR_TEXT_DIM)
            self.screen.blit(dl, (bx + 4, y + 3))
            # Value
            vl = self._font(F_MED).render(f"{int(wt)}", True, col)
            self.screen.blit(vl, (bx + 4, y + 16))
            # Unit
            ul = self._font(F_TINY).render("s", True, COLOUR_TEXT_DIM)
            self.screen.blit(ul, (bx + 4 + vl.get_width() + 2, y + 20))
        y += 44
        return self._panel_divider(y)

    def _panel_utility(self, s, y: int) -> int:
        y = self._panel_section("Utility Scores", y)
        u_ns, u_ew = self.agent.utility_scores
        max_u      = max(u_ns, u_ew, 1.0)
        bar_max    = PANEL_W - 90
        rows = [("U(NS)", u_ns, COLOUR_GREEN_ON),
                ("U(EW)", u_ew, (180, 120, 255))]
        for lbl, val, col in rows:
            pct = min(1.0, val / max_u)
            l   = self._font(F_TINY).render(lbl, True, COLOUR_TEXT_MUTED)
            self.screen.blit(l, (PX, y + 1))
            bar_bg = pygame.Rect(PX + 44, y, bar_max, 10)
            pygame.draw.rect(self.screen, (30, 30, 55), bar_bg, border_radius=3)
            if pct > 0:
                bar_fill = pygame.Rect(PX + 44, y, max(4, int(bar_max * pct)), 10)
                pygame.draw.rect(self.screen, col, bar_fill, border_radius=3)
            vl = self._font(F_TINY).render(f"{val:.1f}", True, COLOUR_TEXT_MUTED)
            self.screen.blit(vl, (PX + 44 + bar_max + 4, y + 1))
            y += 16
        return self._panel_divider(y)

    def _panel_metrics(self, s, y: int) -> int:
        y = self._panel_section("Performance", y)
        w  = (PANEL_W - 24) // 2
        metrics = [
            ("Avg wait",  f"{s.avg_wait_time:.1f}s"),
            ("Cleared",   str(self.sim.total_cleared)),
            ("Cycle",     str(self.sim.cycle)),
            ("Profile",   s.traffic_profile.value.replace("_", " ")),
        ]
        for i, (lbl, val) in enumerate(metrics):
            col = i % 2
            row = i // 2
            bx  = PX + col * w
            by  = y + row * 44
            card = pygame.Rect(bx, by, w - 4, 40)
            pygame.draw.rect(self.screen, (25, 25, 50), card, border_radius=4)
            dl = self._font(F_TINY).render(lbl, True, COLOUR_TEXT_DIM)
            self.screen.blit(dl, (bx + 6, by + 4))
            vl = self._font(F_SMALL).render(val, True, COLOUR_TEXT_PRIMARY)
            self.screen.blit(vl, (bx + 6, by + 20))
        y += 92
        return self._panel_divider(y)

    def _panel_log(self, y: int) -> int:
        y = self._panel_section("Decision Log", y)
        log_font = self._font(F_TINY)
        line_h   = 14
        max_lines = (CANVAS_H - y - 10) // line_h
        lines = self._log_lines[-max_lines:]
        for i, (line, warn) in enumerate(lines):
            col = COLOUR_AMBER_ON if warn else COLOUR_TEXT_DIM
            if i == len(lines) - 1:
                col = COLOUR_TEXT_PRIMARY if not warn else COLOUR_EMERGENCY
            # Truncate to panel width
            max_chars = (PANEL_W - 20) // 6
            display   = line[:max_chars] + ("…" if len(line) > max_chars else "")
            txt = log_font.render(display, True, col)
            self.screen.blit(txt, (PX, y))
            y  += line_h
        return y

    def _add_log(self, line: str, warn: bool = False):
        self._log_lines.append((line, warn))
        if len(self._log_lines) > 60:
            self._log_lines.pop(0)

    # =========================================================================
    # Controls bar
    # =========================================================================

    def _draw_controls(self):
        bar = pygame.Rect(0, CY, WINDOW_WIDTH, CTRL_H)
        pygame.draw.rect(self.screen, COLOUR_PANEL_BG, bar)
        pygame.draw.line(self.screen, COLOUR_PANEL_BORDER,
                         (0, CY), (WINDOW_WIDTH, CY), 1)

        x    = 12
        cy   = CY + CTRL_H // 2
        font = self._font(F_SMALL)

        # Pause / Resume
        label = "Resume" if self.paused else "Pause"
        col   = COLOUR_AMBER_ON if self.paused else COLOUR_GREEN_ON
        self._btn_pause = self._draw_ctrl_btn(label, x, cy, col, font)
        x += self._btn_pause.width + 8

        # Emergency
        self._btn_emrg = self._draw_ctrl_btn(
            "Emergency [E]", x, cy, COLOUR_EMERGENCY, font)
        x += self._btn_emrg.width + 20

        # Profile selector
        lbl = font.render("Profile:", True, COLOUR_TEXT_DIM)
        self.screen.blit(lbl, lbl.get_rect(centery=cy, left=x))
        x  += lbl.get_width() + 6
        self._profile_btns = []
        for profile in TrafficProfile:
            active = profile == self.agent.state.traffic_profile
            col    = COLOUR_GREEN_ON if active else (60, 60, 100)
            short  = profile.value.replace("_RUSH","_R").replace("MORNING","MRN").replace("EVENING","EVN")
            btn    = self._draw_ctrl_btn(short, x, cy, col, self._font(F_TINY))
            self._profile_btns.append((btn, profile))
            x     += btn.width + 4
        x += 14

        # Speed selector
        slbl = font.render("Speed:", True, COLOUR_TEXT_DIM)
        self.screen.blit(slbl, slbl.get_rect(centery=cy, left=x))
        x += slbl.get_width() + 6
        self._speed_btns = []
        for spd in [1, 2, 4, 8]:
            active = spd == self.speed
            col    = COLOUR_GREEN_ON if active else (60, 60, 100)
            btn    = self._draw_ctrl_btn(f"{spd}x", x, cy, col, self._font(F_TINY))
            self._speed_btns.append((btn, spd))
            x     += btn.width + 4

        # Key hints
        hints = font.render("Space=pause  E=emergency  ESC=quit", True, COLOUR_TEXT_DIM)
        self.screen.blit(hints, hints.get_rect(centery=cy, right=WINDOW_WIDTH - 10))

    def _draw_ctrl_btn(self, label, cx, cy, colour, font):
        txt    = font.render(label, True, colour)
        pad    = 10
        rect   = pygame.Rect(cx, cy - 13, txt.get_width() + pad * 2, 26)
        pygame.draw.rect(self.screen, (30, 30, 55), rect, border_radius=4)
        pygame.draw.rect(self.screen, colour, rect, 1, border_radius=4)
        self.screen.blit(txt, txt.get_rect(center=rect.center))
        return rect

    # =========================================================================
    # Font cache
    # =========================================================================

    def _load_fonts(self):
        self._fonts = {}
        for size in [F_TINY, F_SMALL, F_MED, F_LARGE, F_TITLE]:
            try:
                self._fonts[size] = pygame.font.SysFont("monospace", size)
            except Exception:
                self._fonts[size] = pygame.font.Font(None, size + 4)

    def _font(self, size: int) -> pygame.font.Font:
        return self._fonts.get(size, self._fonts[F_SMALL])