# =============================================================================
# gui.py — Pygame Visual Simulation (v2)
# Improvements:
#   - Signal lights on vertical poles, opposite side of road (realistic)
#   - Correct pedestrian crossing logic (NS active = EW crossings light up)
#   - Pedestrian buttons on right panel, no overlap
#   - Animated cars (rectangles) crossing and disappearing
#   - Animated pedestrians (small squares) crossing zebra stripes
#   - Traffic density slider (multiplies arrival rates)
#   - Time-of-day manual override (no auto-revert)
# =============================================================================

import sys, time, random, math
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

# ── Layout ────────────────────────────────────────────────────────────────────
CANVAS_W  = 700
PANEL_W   = WINDOW_WIDTH - CANVAS_W       # 400
CTRL_H    = 52
CANVAS_H  = WINDOW_HEIGHT - CTRL_H

IX = CANVAS_W // 2
IY = CANVAS_H // 2 - 10
ROAD_W    = 88
LANE_W    = ROAD_W // 2

# Signal pole geometry
POLE_H    = 36
POLE_W    = 3
BULB_BOX_W = 16
BULB_BOX_H = 46
BULB_R    = 5

# Crossing geometry
STRIPE_W  = 7
STRIPE_GAP = 4
CROSSING_DEPTH = 14

# Panel
PX        = CANVAS_W + 12
PY_START  = 14
CY        = WINDOW_HEIGHT - CTRL_H

# Font sizes
F_TINY=11; F_SMALL=13; F_MED=15; F_LARGE=18

def lerp_colour(a,b,t):
    return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))

def with_alpha(surf, colour, alpha, rect):
    s = pygame.Surface((rect[2],rect[3]), pygame.SRCALPHA)
    s.fill((*colour, alpha))
    surf.blit(s,(rect[0],rect[1]))


# ── Animated entities ─────────────────────────────────────────────────────────

class Car:
    W, H = 14, 22
    COLOUR = (91,106,255)
    SPEED  = 90   # px/s

    def __init__(self, direction):
        self.direction = direction
        self.alive = True
        # Start position and movement vector
        if direction == NORTH:
            self.x = IX - LANE_W + 12#- self.W//2 
            # self.y = IY + ROAD_W//2 + 4
            # self.x = IX + LANE_W//2 - self.W//2
            self.y = IY - ROAD_W//2 - 4
            self.dx, self.dy = 0, self.SPEED
            self.w, self.h = self.W, self.H
        elif direction == SOUTH:
            self.x = IX + LANE_W//2 - self.W//2
            # self.y = IY - ROAD_W//2 - 4
            # self.x = IX - LANE_W - self.W//2
            self.y = IY + ROAD_W//2 + 4
            self.dx, self.dy = 0, -self.SPEED
            self.w, self.h = self.W, self.H
        elif direction == EAST:
            # self.x = IX - ROAD_W//2 - 4
            self.y = IY - LANE_W +12# - self.W//2
            self.x = IX + ROAD_W//2 + 4
            # self.y = IY + LANE_W//2 - self.W//2
            self.dx, self.dy = -self.SPEED, 0
            self.w, self.h = self.H, self.W
        else:  # WEST
            # self.x = IX + ROAD_W//2 + 4
            self.y = IY + LANE_W//2 - self.W//2
            self.x = IX - ROAD_W//2 - 4
            # self.y = IY - LANE_W - self.W//2
            self.dx, self.dy = self.SPEED, 0
            self.w, self.h = self.H, self.W

    def update(self, dt):
        self.x += self.dx * dt
        self.y += self.dy * dt
        # Kill when off canvas
        if (self.x < -40 or self.x > CANVAS_W+40 or
            self.y < -40 or self.y > CANVAS_H+40):
            self.alive = False

    def draw(self, surf):
        pygame.draw.rect(surf, self.COLOUR,
                         (int(self.x), int(self.y), self.w, self.h), border_radius=2)
        pygame.draw.rect(surf, (140,150,255),
                         (int(self.x), int(self.y), self.w, self.h), 1, border_radius=2)


class Pedestrian:
    SIZE   = 6
    COLOUR = (255, 165, 0)
    SPEED  = 50   # px/s

    def __init__(self, direction):
        # direction = which crossing they use
        # NS active -> pedestrians cross on E or W side (horizontal crossing)
        # EW active -> pedestrians cross on N or S side (vertical crossing)
        self.direction = direction
        self.alive = True
        self.alpha = 255

        if direction == NORTH:  # crossing south side of north road (vertical stripe)
            self.x = float(IX + ROAD_W//2 + 4 + random.randint(2,10))
            self.y = float(IY + ROAD_W//2 + CROSSING_DEPTH)
            self.dx, self.dy = 0, -self.SPEED
        elif direction == SOUTH:  # north side of south road
            self.x = float(IX - ROAD_W//2 - 18 + random.randint(2,10))
            
            self.y = float(IY - ROAD_W//2 - CROSSING_DEPTH)
            self.dx, self.dy = 0, self.SPEED
        elif direction == EAST:  # east side (vertical crossing)
            self.x = float(IX - ROAD_W//2 - CROSSING_DEPTH)
            self.y = float(IY + ROAD_W//2 + 4 + random.randint(2,10))
            self.dx, self.dy = self.SPEED, 0
        else:  # WEST
            
            self.x = float(IX + ROAD_W//2 + CROSSING_DEPTH)
            self.y = float(IY - ROAD_W//2 - 18 + random.randint(2,10))
            self.dx, self.dy = -self.SPEED, 0

        self._dist = 0
        self._total = ROAD_W + 20

    def update(self, dt):
        self.x += self.dx * dt
        self.y += self.dy * dt
        self._dist += (abs(self.dx) + abs(self.dy)) * dt
        if self._dist >= self._total:
            self.alive = False

    def draw(self, surf):
        if not self.alive: return
        col = self.COLOUR
        pygame.draw.rect(surf, col,
                         (int(self.x)-self.SIZE//2,
                          int(self.y)-self.SIZE//2,
                          self.SIZE, self.SIZE), border_radius=1)


# =============================================================================
# TrafficGUI v2
# =============================================================================

class TrafficGUI:
    def __init__(self, simulator, agent, logger):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        self.screen  = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock   = pygame.time.Clock()
        self.sim     = simulator
        self.agent   = agent
        self.logger  = logger
        self._load_fonts()

        self.running       = True
        self.paused        = False
        self.speed         = 2
        self._cycle_accum  = 0.0
        self._last_tick    = time.time()

        # Traffic density multiplier (1.0 = normal, 3.0 = heavy)
        self.density       = 1.0
        # Manual profile override — prevents auto-switching by hour
        self.manual_profile = None

        # Animation
        self._cars: list         = []
        self._peds: list         = []
        self._car_spawn_timer    = [0.0]*4
        self._phase_flash        = 0.0
        self._flash_dir          = 1
        self._queue_anim         = [0.0]*4

        # Log
        self._log_lines: list    = []
        self._prev_decision      = ""

        # Button rects (populated during draw)
        self._btn_pause  = pygame.Rect(0,0,0,0)
        self._btn_emrg   = pygame.Rect(0,0,0,0)
        self._ped_btns   = {}   # direction -> Rect
        self._profile_btns = []
        self._speed_btns   = []
        self._density_rect = pygame.Rect(0,0,0,0)

    # =========================================================================
    # Main loop
    # =========================================================================

    def run(self):
        while self.running:
            dt = self.clock.tick(TARGET_FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()

    # =========================================================================
    # Events
    # =========================================================================

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    self._handle_drag(event.pos)

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:  self.running = False
        elif key == pygame.K_SPACE: self.paused = not self.paused
        elif key == pygame.K_e:     self._trigger_emergency()
        elif key == pygame.K_1:     self.speed = 1
        elif key == pygame.K_2:     self.speed = 2
        elif key == pygame.K_4:     self.speed = 4
        elif key == pygame.K_8:     self.speed = 8

    def _handle_click(self, pos):
        mx,my = pos
        if self._btn_pause.collidepoint(mx,my):
            self.paused = not self.paused
        elif self._btn_emrg.collidepoint(mx,my):
            self._trigger_emergency()
        for d, rect in self._ped_btns.items():
            if rect.collidepoint(mx,my):
                self.sim.press_pedestrian_button(d)
                self._add_log(f"Ped request: {DIRECTION_NAMES[d]} crossing")
        for btn,prof in self._profile_btns:
            if btn.collidepoint(mx,my):
                self.manual_profile = prof
                self.sim.set_profile(prof, lock=True)
                self._add_log(f"Profile -> {prof.value}")
        for btn,spd in self._speed_btns:
            if btn.collidepoint(mx,my):
                self.speed = spd
        # density slider drag
        if self._density_rect.collidepoint(mx,my):
            self._update_density(mx)

    def _handle_drag(self, pos):
        if self._density_rect.collidepoint(pos[0], pos[1]):
            self._update_density(pos[0])

    def _update_density(self, mx):
        r = self._density_rect
        t = max(0.0, min(1.0, (mx - r.x) / r.width))
        self.density = round(1.0 + t * 4.0, 1)   # 1x to 5x

    def _trigger_emergency(self):
        d = random.choice(DIRECTIONS)
        self.sim.trigger_emergency(d)
        self._add_log(f"EMERGENCY from {DIRECTION_NAMES[d]}!", warn=True)

    # =========================================================================
    # Update
    # =========================================================================

    def _update(self, dt):
        if not self.paused:
            # Apply manual profile override before each cycle
            if self.manual_profile is not None:
                if self.sim.profile != self.manual_profile:
                    self.sim.set_profile(self.manual_profile)

            self._cycle_accum += dt * self.speed
            while self._cycle_accum >= 1.0:
                # Apply density to arrival rates temporarily
                self._apply_density()
                run_one_cycle(self.sim, self.agent, self.logger)
                self._cycle_accum -= 1.0
                dec = self.agent.last_decision
                if dec != self._prev_decision:
                    warn = any(w in dec for w in
                               ["EMERGENCY","FORCE","STARVATION","switching"])
                    self._add_log(f"[{self.sim.cycle}] {dec}", warn=warn)
                    self._prev_decision = dec

            # Spawn cars based on queue lengths
            phase = self.agent.state.current_phase
            served = []
            if phase == Phase.NS_GREEN: served = [NORTH, SOUTH]
            elif phase == Phase.EW_GREEN: served = [EAST, WEST]
            for d in DIRECTIONS:
                self._car_spawn_timer[d] -= dt
                if (d in served and
                        self._car_spawn_timer[d] <= 0 and
                        self.agent.state.queue_length[d] > 0):
                    self._cars.append(Car(d))
                    interval = max(0.4, 1.5 / self.density)
                    self._car_spawn_timer[d] = interval

            # Spawn pedestrians on active crossings
            self._maybe_spawn_pedestrian(phase)

        # Update and cull entities
        for c in self._cars: c.update(dt)
        for p in self._peds: p.update(dt)
        self._cars = [c for c in self._cars if c.alive]
        self._peds = [p for p in self._peds if p.alive]

        # Phase pulse animation
        self._phase_flash += self._flash_dir * dt * 2
        if self._phase_flash >= 1.0: self._phase_flash=1.0; self._flash_dir=-1
        elif self._phase_flash <= 0.0: self._phase_flash=0.0; self._flash_dir=1

        # Smooth queue bars
        for d in DIRECTIONS:
            target = self.agent.state.queue_length[d] / 25.0
            self._queue_anim[d] += (target - self._queue_anim[d]) * min(dt*8,1.0)

    def _apply_density(self):
        from config import ARRIVAL_RATES
        rates = ARRIVAL_RATES[self.sim.profile]
        for d in DIRECTIONS:
            self.sim._lanes[d]._base_rate = rates[d] * self.density

    def _maybe_spawn_pedestrian(self, phase):
        # Pedestrians cross perpendicular to vehicle flow
        # NS_GREEN -> pedestrians cross on N crosswalk and S crosswalk
        #   (they are standing at north/south kerb crossing east-west direction)
        # EW_GREEN -> pedestrians cross on E crosswalk and W crosswalk
        if phase == Phase.NS_GREEN:
            crossing_dirs = [NORTH, SOUTH]
        elif phase == Phase.EW_GREEN:
            crossing_dirs = [EAST, WEST]
        else:
            return
        for d in crossing_dirs:
            if (self.agent.state.pedestrian_request[d] and
                    random.random() < 0.02):
                self._peds.append(Pedestrian(d))

    # =========================================================================
    # Draw
    # =========================================================================

    def _draw(self):
        self.screen.fill(COLOUR_BG)
        self._draw_canvas()
        self._draw_panel()
        self._draw_controls()

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _draw_canvas(self):
        s    = self.agent.state
        phase = s.current_phase

        # Roads
        pygame.draw.rect(self.screen, COLOUR_ROAD,
                         (0, IY-ROAD_W//2, CANVAS_W, ROAD_W))
        pygame.draw.rect(self.screen, COLOUR_ROAD,
                         (IX-ROAD_W//2, 0, ROAD_W, CANVAS_H))
        pygame.draw.rect(self.screen, COLOUR_CENTRE,
                         (IX-ROAD_W//2, IY-ROAD_W//2, ROAD_W, ROAD_W))

        # Lane centre dashes
        self._dash_h(IY, 0, IX-ROAD_W//2)
        self._dash_h(IY, IX+ROAD_W//2, CANVAS_W)
        self._dash_v(IX, 0, IY-ROAD_W//2)
        self._dash_v(IX, IY+ROAD_W//2, CANVAS_H)

        # Crossings (correct: NS active = N+S crossings light up = E/W side stripes)
        self._draw_crossings(phase)

        # Queue bars
        self._draw_queues(s)

        # Entities (behind signals)
        for c in self._cars:  c.draw(self.screen)
        for p in self._peds:  p.draw(self.screen)

        # Signal poles (realistic positions)
        self._draw_signals(phase)

        # Direction labels
        self._draw_labels()

        # Emergency overlay
        if phase == Phase.EMERGENCY:
            self._draw_emergency_overlay()

    def _dash_h(self, y, x0, x1):
        x = x0
        while x < x1:
            pygame.draw.line(self.screen, COLOUR_CENTRE,
                             (x,y),(min(x+8,x1),y),1)
            x += 14

    def _dash_v(self, x, y0, y1):
        y = y0
        while y < y1:
            pygame.draw.line(self.screen, COLOUR_CENTRE,
                             (x,y),(x,min(y+8,y1)),1)
            y += 14

    # ── Crossings ─────────────────────────────────────────────────────────────
    # CORRECT LOGIC:
    # NS_GREEN = vehicles flow N+S = pedestrians cross E-W direction
    #   -> the NORTH crossing (people standing N of intersection cross E-W)
    #      is located at the TOP of the intersection (vertical stripes)
    #   -> the SOUTH crossing is at the BOTTOM (vertical stripes)  
    # EW_GREEN = vehicles flow E+W = pedestrians cross N-S direction
    #   -> EAST crossing = right side (horizontal strip
    #   -> WEST crossing = left side (horizontal stripes)

    def _draw_crossings(self, phase):
        ns_active = phase == Phase.EW_GREEN
        ew_active = phase == Phase.NS_GREEN
        s = self.agent.state

        # North crossing — top of intersection, horizontal stripes
        # Active when NS_GREEN (pedestrians from north cross E-W here)
        self._draw_h_crossing(
            IX-ROAD_W//2, IX+ROAD_W//2,
            IY-ROAD_W//2-16, ns_active,
            s.pedestrian_request[NORTH])

        # South crossing — bottom, horizontal stripes
        self._draw_h_crossing(
            IX-ROAD_W//2, IX+ROAD_W//2,
            IY+ROAD_W//2+2, ns_active,
            s.pedestrian_request[SOUTH])

        # East crossing — right side, vertical stripes
        # Active when EW_GREEN (pedestrians from east cross N-S here)
        self._draw_v_crossing(
            IX+ROAD_W//2+2, IY-ROAD_W//2, IY+ROAD_W//2,
            ew_active, s.pedestrian_request[EAST])

        # West crossing — left side, vertical stripes
        self._draw_v_crossing(
            IX-ROAD_W//2-16, IY-ROAD_W//2, IY+ROAD_W//2,
            ew_active, s.pedestrian_request[WEST])

    def _draw_h_crossing(self, x0, x1, y, active, ped_req):
        col   = COLOUR_CROSSING_WALK if active else COLOUR_CROSSING_WAIT
        alpha = 210 if active else 55
        if ped_req and active:
            col   = (255,255,180)
            alpha = 230
        x = x0
        step = STRIPE_W + STRIPE_GAP
        while x < x1:
            w = min(STRIPE_W, x1-x)
            with_alpha(self.screen, col, alpha, (x,y,w,CROSSING_DEPTH))
            x += step

    def _draw_v_crossing(self, x, y0, y1, active, ped_req):
        col   = COLOUR_CROSSING_WALK if active else COLOUR_CROSSING_WAIT
        alpha = 210 if active else 55
        if ped_req and active:
            col   = (255,255,180)
            alpha = 230
        y = y0
        step = STRIPE_W + STRIPE_GAP
        while y < y1:
            h = min(STRIPE_W, y1-y)
            with_alpha(self.screen, col, alpha, (x,y,CROSSING_DEPTH,h))
            y += step

    # ── Queue bars ────────────────────────────────────────────────────────────

    def _draw_queues(self, state):
        MAX_LEN = 110
        THICK   = 16
        for d in DIRECTIONS:
            length = int(self._queue_anim[d] * MAX_LEN)
            col    = COLOUR_QUEUE_NS if d in (NORTH,SOUTH) else COLOUR_QUEUE_EW
            count  = state.queue_length[d]
            if length < 2 and count == 0: continue
            if d == NORTH:
                rect = (IX-8-THICK, IY-ROAD_W//2-length, THICK, max(length,2))
            elif d == SOUTH:
                rect = (IX+8, IY+ROAD_W//2, THICK, max(length,2))
            elif d == EAST:
                rect = (IX+ROAD_W//2, IY-8-THICK, max(length,2), THICK)
            else:
                rect = (IX-ROAD_W//2-length, IY+8, max(length,2), THICK)
            with_alpha(self.screen, col, 150, rect)
            lbl = self._font(F_TINY).render(str(count), True, col)
            if d == NORTH:   self.screen.blit(lbl,(rect[0]+2,rect[1]-13))
            elif d == SOUTH: self.screen.blit(lbl,(rect[0]+2,rect[1]+rect[3]+2))
            elif d == EAST:  self.screen.blit(lbl,(rect[0]+rect[2]+2,rect[1]+2))
            else:            self.screen.blit(lbl,(rect[0]-13,rect[1]+2))

    # ── Signal lights on poles ────────────────────────────────────────────────
    # Each signal is on the OPPOSITE side of the road from the approaching lane,
    # facing the oncoming driver.
    #
    # NORTH approach: driver comes from top, signal is on SOUTH side of that road
    #   -> placed just above intersection top edge, centred on left lane (N-bound)
    # SOUTH approach: driver comes from bottom, signal on NORTH side of south road
    #   -> placed just below intersection bottom edge, right lane (S-bound)
    # EAST approach: driver comes from right, signal on WEST side of east road
    #   -> placed just right of intersection right edge, top lane (E-bound)
    # WEST approach: driver comes from left, signal on EAST side of west road
    #   -> placed just left of intersection left edge, bottom lane (W-bound)

    def _draw_signals(self, phase):
        sig = self._phase_to_signals(phase)
        gap = 2

        # Right-hand-road layout: each signal sits on the driver's right side,
        # outside the carriageway, at the far corner of the approach.

        # North signal: top-right corner, outside the northbound approach.
        nx = IX + ROAD_W//2 + gap + BULB_BOX_W//2
        ny = IY - ROAD_W//2 - gap - POLE_H - BULB_BOX_H
        self._draw_signal_v(nx, ny, *sig[NORTH])

        # South signal: bottom-left corner, outside the southbound approach.
        sx = IX - ROAD_W//2 - gap - BULB_BOX_W//2
        sy = IY + ROAD_W//2 + gap
        self._draw_signal_v(sx, sy, *sig[SOUTH])

        # East signal: bottom-right corner, outside the eastbound approach.
        ex = IX + ROAD_W//2 + POLE_H + gap
        ey = IY + ROAD_W//2 + gap + BULB_BOX_W//2
        self._draw_signal_h(ex, ey, *sig[EAST])

        # West signal: top-left corner, outside the westbound approach.
        wx = IX - ROAD_W//2 - BULB_BOX_H - gap
        wy = IY - ROAD_W//2 - gap - BULB_BOX_W//2
        self._draw_signal_h(wx, wy, *sig[WEST])

    def _draw_signal_v(self, cx, y, r, a, g):
        # Vertical signal box (tall)
        box = pygame.Rect(cx - BULB_BOX_W//2, y, BULB_BOX_W, BULB_BOX_H)
        pygame.draw.rect(self.screen, (15,15,38), box, border_radius=3)
        pygame.draw.rect(self.screen, (50,50,80), box, 1, border_radius=3)
        # Pole below
        px = cx - POLE_W//2
        py = y + BULB_BOX_H
        pygame.draw.rect(self.screen, (60,60,90), (px, py, POLE_W, POLE_H))
        # Bulbs
        spacing = BULB_BOX_H // 4
        for i,(on,col) in enumerate([(r,COLOUR_RED_ON),(a,COLOUR_AMBER_ON),(g,COLOUR_GREEN_ON)]):
            by = y + spacing*(i+1)
            pygame.draw.circle(self.screen, col if on else COLOUR_BULB_OFF, (cx, by), BULB_R)

    def _draw_signal_h(self, x, cy, r, a, g):
        # Horizontal signal box (wide)
        box = pygame.Rect(x, cy - BULB_BOX_W//2, BULB_BOX_H, BULB_BOX_W)
        pygame.draw.rect(self.screen, (15,15,38), box, border_radius=3)
        pygame.draw.rect(self.screen, (50,50,80), box, 1, border_radius=3)
        # Pole
        pygame.draw.rect(self.screen, (60,60,90),
                         (x-POLE_H, cy-POLE_W//2, POLE_H, POLE_W))
        # Bulbs
        spacing = BULB_BOX_H // 4
        for i,(on,col) in enumerate([(r,COLOUR_RED_ON),(a,COLOUR_AMBER_ON),(g,COLOUR_GREEN_ON)]):
            bx = x + spacing*(i+1)
            pygame.draw.circle(self.screen, col if on else COLOUR_BULB_OFF, (bx, cy), BULB_R)

    def _phase_to_signals(self, phase):
        G = (False,False,True)
        A = (False,True, False)
        R = (True, False,False)
        if phase == Phase.NS_GREEN:
            return {NORTH:G,SOUTH:G,EAST:R,WEST:R}
        elif phase == Phase.NS_YELLOW:
            return {NORTH:A,SOUTH:A,EAST:R,WEST:R}
        elif phase == Phase.EW_GREEN:
            return {NORTH:R,SOUTH:R,EAST:G,WEST:G}
        elif phase == Phase.EW_YELLOW:
            return {NORTH:R,SOUTH:R,EAST:A,WEST:A}
        else:
            return {d:R for d in DIRECTIONS}

    # ── Labels ────────────────────────────────────────────────────────────────

    def _draw_labels(self):
        font = self._font(F_TINY)
        labels = {NORTH:(IX,18), SOUTH:(IX,CANVAS_H-18),
                  EAST:(CANVAS_W-22,IY), WEST:(22,IY)}
        arrows = {NORTH:"^ N", SOUTH:"v S", EAST:"E >", WEST:"< W"}
        for d,(x,y) in labels.items():
            t = font.render(arrows[d], True, COLOUR_TEXT_DIM)
            self.screen.blit(t, t.get_rect(center=(x,y)))

    # ── Emergency overlay ──────────────────────────────────────────────────────

    def _draw_emergency_overlay(self):
        pulse = abs(self._phase_flash-0.5)*2
        with_alpha(self.screen, COLOUR_EMERGENCY, int(25+pulse*40),
                   (0,0,CANVAS_W,CANVAS_H))
        font = self._font(F_SMALL)
        txt  = font.render("[!] EMERGENCY OVERRIDE - ALL RED", True, COLOUR_EMERGENCY)
        self.screen.blit(txt, txt.get_rect(center=(CANVAS_W//2, 36)))

    # =========================================================================
    # Panel
    # =========================================================================

    def _draw_panel(self):
        panel = pygame.Rect(CANVAS_W, 0, PANEL_W, CANVAS_H)
        pygame.draw.rect(self.screen, COLOUR_PANEL_BG, panel)
        pygame.draw.line(self.screen, COLOUR_PANEL_BORDER,
                         (CANVAS_W,0),(CANVAS_W,CANVAS_H),1)
        s = self.agent.state
        y = PY_START
        y = self._panel_phase(s, y)
        y = self._panel_queues(s, y)
        y = self._panel_waits(s, y)
        y = self._panel_utility(s, y)
        y = self._panel_ped_buttons(s, y)
        y = self._panel_metrics(s, y)
        y = self._panel_log(y)

    def _sec(self, title, y):
        t = self._font(F_TINY).render(title.upper(), True, COLOUR_TEXT_DIM)
        self.screen.blit(t,(PX,y))
        return y+15

    def _div(self, y):
        pygame.draw.line(self.screen, COLOUR_PANEL_BORDER,
                         (PX-4,y+4),(WINDOW_WIDTH-8,y+4),1)
        return y+11

    def _panel_phase(self, s, y):
        y = self._sec("Current phase", y)
        phase_col = {
            Phase.NS_GREEN:  ((0,50,25),    COLOUR_GREEN_ON),
            Phase.NS_YELLOW: ((45,30,0),    COLOUR_AMBER_ON),
            Phase.EW_GREEN:  ((18,0,50),    (180,120,255)),
            Phase.EW_YELLOW: ((45,30,0),    COLOUR_AMBER_ON),
            Phase.EMERGENCY: ((55,0,0),     COLOUR_EMERGENCY),
        }
        bg,fg = phase_col.get(s.current_phase,((30,30,50),COLOUR_TEXT_PRIMARY))
        badge = pygame.Rect(PX,y,PANEL_W-24,26)
        pygame.draw.rect(self.screen,bg,badge,border_radius=4)
        txt = self._font(F_MED).render(s.current_phase.name,True,fg)
        self.screen.blit(txt,txt.get_rect(center=badge.center))
        y+=30
        is_y  = s.current_phase in (Phase.NS_YELLOW,Phase.EW_YELLOW)
        max_t = s.yellow_duration if is_y else s.max_green
        pct   = min(1.0, s.phase_elapsed/max(max_t,1))
        bw    = PANEL_W-24
        pygame.draw.rect(self.screen,(35,35,65),(PX,y,bw,5),border_radius=2)
        if pct>0:
            fc = COLOUR_AMBER_ON if pct>0.8 else COLOUR_GREEN_ON
            pygame.draw.rect(self.screen,fc,(PX,y,max(3,int(bw*pct)),5),border_radius=2)
        tl = self._font(F_TINY).render(f"{s.phase_elapsed}s / {max_t}s",True,COLOUR_TEXT_MUTED)
        self.screen.blit(tl,(PX,y+8))
        return self._div(y+22)

    def _panel_queues(self, s, y):
        y = self._sec("Queue lengths", y)
        bmax = PANEL_W-80
        dirs = ["N","S","E","W"]
        cols = [COLOUR_QUEUE_NS,COLOUR_QUEUE_NS,COLOUR_QUEUE_EW,COLOUR_QUEUE_EW]
        for d in DIRECTIONS:
            pct = min(1.0,s.queue_length[d]/25.0)
            l = self._font(F_TINY).render(dirs[d],True,COLOUR_TEXT_MUTED)
            self.screen.blit(l,(PX,y+1))
            pygame.draw.rect(self.screen,(28,28,52),(PX+14,y,bmax,9),border_radius=2)
            if pct>0:
                pygame.draw.rect(self.screen,cols[d],(PX+14,y,max(3,int(bmax*pct)),9),border_radius=2)
            c = self._font(F_TINY).render(str(s.queue_length[d]),True,COLOUR_TEXT_MUTED)
            self.screen.blit(c,(PX+14+bmax+3,y+1))
            y+=15
        return self._div(y)

    def _panel_waits(self, s, y):
        y = self._sec("Wait times", y)
        dirs=["N","S","E","W"]
        w=(PANEL_W-24)//4
        for d in DIRECTIONS:
            bx=PX+d*w; wt=s.wait_time[d]
            col=(COLOUR_EMERGENCY if wt>60 else
                 COLOUR_AMBER_ON  if wt>30 else COLOUR_GREEN_ON)
            card=pygame.Rect(bx,y,w-3,36)
            pygame.draw.rect(self.screen,(22,22,48),card,border_radius=3)
            dl=self._font(F_TINY).render(dirs[d],True,COLOUR_TEXT_DIM)
            self.screen.blit(dl,(bx+4,y+3))
            vl=self._font(F_MED).render(str(int(wt)),True,col)
            self.screen.blit(vl,(bx+4,y+15))
            ul=self._font(F_TINY).render("s",True,COLOUR_TEXT_DIM)
            self.screen.blit(ul,(bx+4+vl.get_width()+1,y+19))
        return self._div(y+42)

    def _panel_utility(self, s, y):
        y = self._sec("Utility scores", y)
        u_ns,u_ew = self.agent.utility_scores
        maxu = max(u_ns,u_ew,1.0)
        bmax = PANEL_W-88
        for lbl,val,col in [("U(NS)",u_ns,COLOUR_GREEN_ON),
                             ("U(EW)",u_ew,(180,120,255))]:
            pct=min(1.0,val/maxu)
            l=self._font(F_TINY).render(lbl,True,COLOUR_TEXT_MUTED)
            self.screen.blit(l,(PX,y+1))
            pygame.draw.rect(self.screen,(28,28,52),(PX+42,y,bmax,9),border_radius=2)
            if pct>0:
                pygame.draw.rect(self.screen,col,(PX+42,y,max(3,int(bmax*pct)),9),border_radius=2)
            vl=self._font(F_TINY).render(f"{val:.0f}",True,COLOUR_TEXT_MUTED)
            self.screen.blit(vl,(PX+42+bmax+3,y+1))
            y+=15
        return self._div(y)

    def _panel_ped_buttons(self, s, y):
        y = self._sec("Pedestrian requests", y)
        font = self._font(F_TINY)
        # 4 buttons, 2 per row
        # NS button = press for NS lane (North/South vehicles)
        # EW button = press for EW lane (East/West vehicles)
        btn_defs = [
            (NORTH, "N crossing",  "NS lane"),
            (SOUTH, "S crossing",  "NS lane"),
            (EAST,  "E crossing",  "EW lane"),
            (WEST,  "W crossing",  "EW lane"),
        ]
        bw = (PANEL_W-28)//2
        bh = 30
        self._ped_btns = {}
        for i,(d,label,sublabel) in enumerate(btn_defs):
            col = i%2; row = i//2
            bx  = PX + col*(bw+4)
            by  = y + row*(bh+4)
            req = s.pedestrian_request[d]
            bg  = (0,45,20) if req else (22,22,48)
            bc  = COLOUR_CROSSING_WALK if req else (60,60,90)
            rect = pygame.Rect(bx,by,bw,bh)
            self._ped_btns[d] = rect
            pygame.draw.rect(self.screen,bg,rect,border_radius=4)
            pygame.draw.rect(self.screen,bc,rect,1,border_radius=4)
            lt = font.render(label,True,
                             COLOUR_CROSSING_WALK if req else COLOUR_TEXT_MUTED)
            self.screen.blit(lt,(bx+5,by+4))
            st = self._font(10).render(sublabel,True,COLOUR_TEXT_DIM)
            self.screen.blit(st,(bx+5,by+16))
        y += 2*(bh+4)+2
        # Explanation note
        note = font.render("Click to request crossing",True,COLOUR_TEXT_DIM)
        self.screen.blit(note,(PX,y))
        y+=14
        return self._div(y)

    def _panel_metrics(self, s, y):
        y = self._sec("Performance", y)
        w=(PANEL_W-24)//2
        metrics=[
            ("Avg wait",  f"{s.avg_wait_time:.1f}s"),
            ("Cleared",   str(self.sim.total_cleared)),
            ("Cycle",     str(self.sim.cycle)),
            ("Density",   f"{self.density:.1f}x"),
        ]
        for i,(lbl,val) in enumerate(metrics):
            bx=PX+(i%2)*w; by=y+(i//2)*42
            card=pygame.Rect(bx,by,w-3,38)
            pygame.draw.rect(self.screen,(22,22,48),card,border_radius=3)
            dl=self._font(F_TINY).render(lbl,True,COLOUR_TEXT_DIM)
            self.screen.blit(dl,(bx+5,by+4))
            vl=self._font(F_SMALL).render(val,True,COLOUR_TEXT_PRIMARY)
            self.screen.blit(vl,(bx+5,by+18))
        return self._div(y+88)

    def _panel_log(self, y):
        y = self._sec("Decision log", y)
        lf   = self._font(F_TINY)
        lh   = 13
        ml   = (CANVAS_H-y-8)//lh
        lines= self._log_lines[-ml:]
        for i,(line,warn) in enumerate(lines):
            col=(COLOUR_AMBER_ON if warn else COLOUR_TEXT_DIM)
            if i==len(lines)-1:
                col=COLOUR_EMERGENCY if warn else COLOUR_TEXT_PRIMARY
            mc=(PANEL_W-18)//6
            disp=line[:mc]+("~" if len(line)>mc else "")
            t=lf.render(disp,True,col)
            self.screen.blit(t,(PX,y))
            y+=lh
        return y

    def _add_log(self,line,warn=False):
        self._log_lines.append((line,warn))
        if len(self._log_lines)>60: self._log_lines.pop(0)

    # =========================================================================
    # Controls bar
    # =========================================================================

    def _draw_controls(self):
        pygame.draw.rect(self.screen,COLOUR_PANEL_BG,
                         (0,CY,WINDOW_WIDTH,CTRL_H))
        pygame.draw.line(self.screen,COLOUR_PANEL_BORDER,
                         (0,CY),(WINDOW_WIDTH,CY),1)
        x=10; cy=CY+CTRL_H//2
        font=self._font(F_SMALL)

        # Pause
        self._btn_pause=self._ctrl_btn(
            "Resume" if self.paused else "Pause",
            x,cy,COLOUR_AMBER_ON if self.paused else COLOUR_GREEN_ON,font)
        x+=self._btn_pause.width+8

        # Emergency
        self._btn_emrg=self._ctrl_btn(
            "Emergency [E]",x,cy,COLOUR_EMERGENCY,font)
        x+=self._btn_emrg.width+12

        # Profile
        lbl=font.render("Profile:",True,COLOUR_TEXT_DIM)
        self.screen.blit(lbl,lbl.get_rect(centery=cy,left=x))
        x+=lbl.get_width()+5
        self._profile_btns=[]
        prof_labels = {
            TrafficProfile.MORNING_RUSH:"Morn Rush",
            TrafficProfile.MIDDAY:"Midday",
            TrafficProfile.EVENING_RUSH:"Eve Rush",
            TrafficProfile.NIGHT:"Night",
        }
        for prof in TrafficProfile:
            active=(self.manual_profile or self.agent.state.traffic_profile)==prof
            col=COLOUR_GREEN_ON if active else (55,55,95)
            btn=self._ctrl_btn(prof_labels[prof],x,cy,col,self._font(F_TINY))
            self._profile_btns.append((btn,prof))
            x+=btn.width+3
        x+=10

        # Speed
        sl=font.render("Spd:",True,COLOUR_TEXT_DIM)
        self.screen.blit(sl,sl.get_rect(centery=cy,left=x))
        x+=sl.get_width()+4
        self._speed_btns=[]
        for spd in [1,2,4,8]:
            active=spd==self.speed
            col=COLOUR_GREEN_ON if active else (55,55,95)
            btn=self._ctrl_btn(f"{spd}x",x,cy,col,self._font(F_TINY))
            self._speed_btns.append((btn,spd))
            x+=btn.width+3
        x+=10

        # Density slider
        dl=font.render("Traffic:",True,COLOUR_TEXT_DIM)
        self.screen.blit(dl,dl.get_rect(centery=cy,left=x))
        x+=dl.get_width()+5
        slider_w=80
        self._density_rect=pygame.Rect(x,cy-5,slider_w,10)
        pygame.draw.rect(self.screen,(40,40,70),self._density_rect,border_radius=4)
        fill_w=int(slider_w*(self.density-1)/4)
        if fill_w>0:
            pygame.draw.rect(self.screen,COLOUR_GREEN_ON,
                             (x,cy-5,fill_w,10),border_radius=4)
        thumb_x=x+fill_w
        pygame.draw.circle(self.screen,COLOUR_TEXT_PRIMARY,(thumb_x,cy),6)
        dv=font.render(f"{self.density:.1f}x",True,COLOUR_TEXT_MUTED)
        self.screen.blit(dv,dv.get_rect(centery=cy,left=x+slider_w+5))

    def _ctrl_btn(self,label,cx,cy,colour,font):
        txt=font.render(label,True,colour)
        pad=8
        rect=pygame.Rect(cx,cy-12,txt.get_width()+pad*2,24)
        pygame.draw.rect(self.screen,(28,28,52),rect,border_radius=3)
        pygame.draw.rect(self.screen,colour,rect,1,border_radius=3)
        self.screen.blit(txt,txt.get_rect(center=rect.center))
        return rect

    # =========================================================================
    # Font cache
    # =========================================================================

    def _load_fonts(self):
        self._fonts={}
        for size in [10,F_TINY,F_SMALL,F_MED,F_LARGE]:
            try:    self._fonts[size]=pygame.font.SysFont("monospace",size)
            except: self._fonts[size]=pygame.font.Font(None,size+4)

    def _font(self,size):
        return self._fonts.get(size,self._fonts[F_SMALL])