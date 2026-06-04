# =============================================================================
# evaluate.py — Phase 5: Evaluation & Comparison
# Runs three experiments:
#   1. Agent vs Fixed-timer baseline across all traffic profiles
#   2. Agent performance under increasing traffic density
#   3. Emergency response time measurement
# Generates matplotlib charts saved to /evaluation_charts/
# =============================================================================

import os, csv, random, math
from dataclasses import dataclass, field
from typing import List, Dict

from config import (
    Phase, TrafficProfile,
    DIRECTIONS, NORTH, SOUTH, EAST, WEST,
    ARRIVAL_RATES, PHASE_DURATIONS, YELLOW_DURATION,
    INITIAL_PHASE,
)
from simulator import IntersectionSimulator
from agent    import TrafficLightAgent
from logger   import SimulationLogger

OUTPUT_DIR = "evaluation_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CYCLES_PER_RUN = 600    # 600 cycles = 10 simulated minutes per run
SEEDS          = [42, 77, 123, 256, 512]   # 5 seeds for averaging


# ── Fixed-timer baseline agent ────────────────────────────────────────────────

class FixedTimerAgent:
    """
    Naive baseline: alternates NS and EW green on a fixed schedule.
    No utility function — purely time-driven.
    Has basic emergency handling: immediately goes all-red on detection
    (no yellow clearance — this is the key difference from the intelligent agent).
    """
    FIXED_GREEN  = 30
    FIXED_YELLOW = 3

    def __init__(self):
        self.current_phase   = Phase.NS_GREEN
        self.phase_elapsed   = 0
        self.avg_wait_time   = 0.0
        self.last_decision   = "fixed timer"
        self._wait_time      = [0.0]*4
        self._cycle          = 0
        self._in_emergency   = False

    def run_cycle(self, reading):
        from config import PHASE_SERVES, GREEN_TO_YELLOW, YELLOW_TO_GREEN
        self._cycle       += 1
        self.phase_elapsed += 1

        # Emergency: go ALL-RED immediately (no yellow clearance)
        if reading.emergency_active:
            if not self._in_emergency:
                self._in_emergency = True
                self.current_phase = Phase.EMERGENCY
                self.phase_elapsed = 0
            return self.current_phase
        elif self._in_emergency:
            self._in_emergency = False
            self.current_phase = Phase.NS_GREEN
            self.phase_elapsed = 0

        # Update wait times
        served = PHASE_SERVES.get(self.current_phase, [])
        for d in DIRECTIONS:
            if d in served: self._wait_time[d] = 0.0
            else:           self._wait_time[d] += 1.0

        # Fixed switching
        if self.current_phase in (Phase.NS_YELLOW, Phase.EW_YELLOW):
            if self.phase_elapsed >= self.FIXED_YELLOW:
                self.current_phase = YELLOW_TO_GREEN[self.current_phase]
                self.phase_elapsed = 0
        elif self.current_phase != Phase.EMERGENCY:
            if self.phase_elapsed >= self.FIXED_GREEN:
                self.current_phase = GREEN_TO_YELLOW[self.current_phase]
                self.phase_elapsed = 0

        total_wait = sum(self._wait_time) / 4
        self.avg_wait_time = self.avg_wait_time*0.95 + total_wait*0.05
        return self.current_phase


# ── Metric collection ─────────────────────────────────────────────────────────

@dataclass
class RunResult:
    profile:       str
    seed:          int
    agent_type:    str
    avg_wait:      float
    max_wait:      float
    throughput:    float
    phase_dist:    Dict[str, float] = field(default_factory=dict)
    emergency_resp: float = 0.0


def run_simulation(profile: TrafficProfile, seed: int,
                   use_agent: bool, density: float = 1.0,
                   inject_emergency: bool = False) -> RunResult:
    """Run one complete simulation and return metrics."""
    from config import ARRIVAL_RATES
    sim   = IntersectionSimulator(
        initial_profile=profile, initial_phase=INITIAL_PHASE,
        seed=seed, start_hour=9.0)
    sim.profile_locked = True

    # Scale arrival rates by density
    for d in DIRECTIONS:
        sim._lanes[d]._base_rate = ARRIVAL_RATES[profile][d] * density

    if use_agent:
        ctrl = TrafficLightAgent(initial_phase=INITIAL_PHASE,
                                 initial_profile=profile)
    else:
        ctrl = FixedTimerAgent()

    wait_history   = []
    cleared_history= []
    phase_counts   = {p.name:0 for p in Phase}
    emrg_response  = []
    emrg_start     = None

    for cycle in range(CYCLES_PER_RUN):
        # Inject one emergency at cycle 200
        if inject_emergency and cycle == 200:
            sim.trigger_emergency(EAST)
            emrg_start = cycle

        reading = sim.get_sensor_reading()
        phase   = ctrl.run_cycle(reading)
        sim.apply_phase(phase)
        sim.step()

        phase_counts[phase.name] += 1
        if hasattr(ctrl, 'state'):
            raw_wait = sum(ctrl.state.wait_time) / 4
        else:
            raw_wait = sum(ctrl._wait_time) / 4
        wait_history.append(raw_wait)
        cleared_history.append(sim.total_cleared)

        # Measure emergency response (cycles until EMERGENCY phase reached)
        if inject_emergency and emrg_start and emrg_response == []:
            if phase == Phase.EMERGENCY:
                emrg_response.append(cycle - emrg_start)

    total = sum(phase_counts.values())
    dist  = {k: round(v/total*100,1) for k,v in phase_counts.items() if v>0}

    cleared_per_cycle = sim.total_cleared / CYCLES_PER_RUN
    max_w = max(wait_history) if wait_history else 0.0

    return RunResult(
        profile       = profile.value,
        seed          = seed,
        agent_type    = "Agent" if use_agent else "Fixed timer",
        avg_wait      = round(sum(wait_history)/len(wait_history), 3),
        max_wait      = round(max_w, 2),
        throughput    = round(cleared_per_cycle, 3),
        phase_dist    = dist,
        emergency_resp= emrg_response[0] if emrg_response else 0,
    )


def avg_results(results: List[RunResult]) -> dict:
    """Average metrics across multiple seeds."""
    return {
        "avg_wait":   round(sum(r.avg_wait   for r in results)/len(results), 3),
        "max_wait":   round(sum(r.max_wait   for r in results)/len(results), 2),
        "throughput": round(sum(r.throughput for r in results)/len(results), 3),
    }


# ── Experiment 1: Agent vs Fixed across profiles ──────────────────────────────

def experiment_1():
    print("\n== Experiment 1: Agent vs Fixed-timer (all profiles) ==")
    profiles = list(TrafficProfile)
    agent_avgs, fixed_avgs = [], []
    agent_tput, fixed_tput = [], []

    for profile in profiles:
        print(f"  {profile.value}...", end=" ", flush=True)
        a_res = [run_simulation(profile,s,True)  for s in SEEDS]
        f_res = [run_simulation(profile,s,False) for s in SEEDS]
        a = avg_results(a_res)
        f = avg_results(f_res)
        agent_avgs.append(a["avg_wait"])
        fixed_avgs.append(f["avg_wait"])
        agent_tput.append(a["throughput"])
        fixed_tput.append(f["throughput"])
        print(f"Agent wait={a['avg_wait']}s  Fixed wait={f['avg_wait']}s")

    return profiles, agent_avgs, fixed_avgs, agent_tput, fixed_tput


# ── Experiment 2: Density stress test ────────────────────────────────────────

def experiment_2():
    print("\n== Experiment 2: Traffic density stress test ==")
    densities  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    profile    = TrafficProfile.MIDDAY
    a_waits, f_waits = [], []
    a_tputs, f_tputs = [], []

    for d in densities:
        print(f"  density={d}x...", end=" ", flush=True)
        a_res = [run_simulation(profile,s,True, density=d)  for s in SEEDS[:3]]
        f_res = [run_simulation(profile,s,False,density=d) for s in SEEDS[:3]]
        a = avg_results(a_res)
        f = avg_results(f_res)
        a_waits.append(a["avg_wait"])
        f_waits.append(f["avg_wait"])
        a_tputs.append(a["throughput"])
        f_tputs.append(f["throughput"])
        print(f"Agent={a['avg_wait']}s  Fixed={f['avg_wait']}s")

    return densities, a_waits, f_waits, a_tputs, f_tputs


# ── Experiment 3: Emergency response ─────────────────────────────────────────

def experiment_3():
    print("\n== Experiment 3: Emergency response timing ==")
    profiles = list(TrafficProfile)
    a_resp, f_resp = [], []

    for profile in profiles:
        print(f"  {profile.value}...", end=" ", flush=True)
        a_res = [run_simulation(profile,s,True, inject_emergency=True)
                 for s in SEEDS]
        f_res = [run_simulation(profile,s,False,inject_emergency=True)
                 for s in SEEDS]
        a_avg = round(sum(r.emergency_resp for r in a_res)/len(a_res),1)
        f_avg = round(sum(r.emergency_resp for r in f_res)/len(f_res),1)
        a_resp.append(a_avg)
        f_resp.append(f_avg)
        print(f"Agent={a_avg} cycles  Fixed={f_avg} cycles")

    return profiles, a_resp, f_resp


# ── Chart generation ──────────────────────────────────────────────────────────

def make_charts(exp1, exp2, exp3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("\n[!] matplotlib not installed — skipping charts")
        print("    Run: pip install matplotlib")
        return

    plt.rcParams.update({
        "font.family":  "monospace",
        "font.size":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "#F8F8F6",
        "axes.grid":          True,
        "grid.color":         "#DDDDDD",
        "grid.linewidth":     0.5,
    })

    C_AGENT  = "#1D9E75"   # teal
    C_FIXED  = "#E24B4A"   # red
    C_AGENT2 = "#5DCAA5"   # light teal
    C_FIXED2 = "#F09595"   # light red

    profiles, a_wait, f_wait, a_tput, f_tput = exp1
    densities, da_wait, df_wait, da_tput, df_tput = exp2
    profiles3, a_resp, f_resp = exp3
    pnames = [p.value.replace("_"," ") for p in profiles]

    # ── Chart 1: Avg wait by profile ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Experiment 1 — Agent vs Fixed-timer across traffic profiles",
                 fontsize=11, fontweight="bold", y=1.01)

    ax = axes[0]
    x  = range(len(pnames))
    w  = 0.35
    bars_a = ax.bar([i-w/2 for i in x], a_wait, w,
                    label="Intelligent agent", color=C_AGENT,  zorder=3)
    bars_f = ax.bar([i+w/2 for i in x], f_wait, w,
                    label="Fixed timer",       color=C_FIXED,  zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(pnames, rotation=12, ha="right")
    ax.set_ylabel("Average wait time (seconds)")
    ax.set_title("Average wait time per profile")
    ax.legend()
    for bar in bars_a:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=8, color=C_AGENT)
    for bar in bars_f:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=8, color=C_FIXED)

    ax2 = axes[1]
    bars_a2 = ax2.bar([i-w/2 for i in x], a_tput, w,
                      label="Intelligent agent", color=C_AGENT,  zorder=3)
    bars_f2 = ax2.bar([i+w/2 for i in x], f_tput, w,
                      label="Fixed timer",       color=C_FIXED,  zorder=3)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(pnames, rotation=12, ha="right")
    ax2.set_ylabel("Throughput (vehicles / cycle)")
    ax2.set_title("Throughput per profile")
    ax2.legend()

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "exp1_profile_comparison.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path1}")

    # ── Chart 2: Density stress test ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Experiment 2 — Performance under increasing traffic density",
                 fontsize=11, fontweight="bold", y=1.01)

    ax3 = axes[0]
    ax3.plot(densities, da_wait, "o-", color=C_AGENT, label="Intelligent agent",
             linewidth=2, markersize=6)
    ax3.plot(densities, df_wait, "s-", color=C_FIXED, label="Fixed timer",
             linewidth=2, markersize=6)
    ax3.fill_between(densities, da_wait, df_wait, alpha=0.12,
                     color=C_AGENT, label="Improvement gap")
    ax3.set_xlabel("Traffic density multiplier")
    ax3.set_ylabel("Average wait time (seconds)")
    ax3.set_title("Wait time vs traffic density")
    ax3.legend()

    ax4 = axes[1]
    ax4.plot(densities, da_tput, "o-", color=C_AGENT, label="Intelligent agent",
             linewidth=2, markersize=6)
    ax4.plot(densities, df_tput, "s-", color=C_FIXED, label="Fixed timer",
             linewidth=2, markersize=6)
    ax4.set_xlabel("Traffic density multiplier")
    ax4.set_ylabel("Throughput (vehicles / cycle)")
    ax4.set_title("Throughput vs traffic density")
    ax4.legend()

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "exp2_density_stress.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    # ── Chart 3: Emergency response ───────────────────────────────────────────
    fig, ax5 = plt.subplots(figsize=(8, 4))
    fig.suptitle("Experiment 3 — Emergency response time (cycles to reach all-red)",
                 fontsize=11, fontweight="bold")

    x = range(len(pnames))
    bars_ae = ax5.bar([i-w/2 for i in x], a_resp, w,
                      label="Intelligent agent", color=C_AGENT,  zorder=3)
    bars_fe = ax5.bar([i+w/2 for i in x], f_resp, w,
                      label="Fixed timer",       color=C_FIXED,  zorder=3)
    ax5.set_xticks(list(x)); ax5.set_xticklabels(pnames, rotation=12, ha="right")
    ax5.set_ylabel("Cycles to emergency all-red")
    ax5.set_title("The intelligent agent always runs yellow clearance first\n"
                  "Fixed timer goes all-red immediately (if emergency logic is present)")
    ax5.legend()
    for bar in bars_ae:
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{bar.get_height():.0f}", ha="center", va="bottom",
                 fontsize=8, color=C_AGENT)

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "exp3_emergency_response.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")

    # ── Chart 4: Summary improvement table ────────────────────────────────────
    fig, ax6 = plt.subplots(figsize=(10, 3))
    ax6.axis("off")
    fig.suptitle("Summary — Agent vs Fixed-timer improvement",
                 fontsize=11, fontweight="bold")

    improvements = []
    for i,p in enumerate(pnames):
        wait_imp = round((f_wait[i]-a_wait[i])/max(f_wait[i],0.01)*100, 1)
        tput_imp = round((a_tput[i]-f_tput[i])/max(f_tput[i],0.01)*100, 1)
        improvements.append([p,
                              f"{a_wait[i]:.2f}s",
                              f"{f_wait[i]:.2f}s",
                              f"{wait_imp:+.1f}%",
                              f"{a_tput[i]:.3f}",
                              f"{f_tput[i]:.3f}",
                              f"{tput_imp:+.1f}%"])

    cols = ["Profile","Agent wait","Fixed wait","Wait improv.",
            "Agent tput","Fixed tput","Tput improv."]
    tbl = ax6.table(cellText=improvements, colLabels=cols,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Colour header row
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#1D9E75")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    # Colour improvement cells
    for i in range(1, len(improvements)+1):
        for j in [3,6]:
            val = float(improvements[i-1][j].replace("%","").replace("+",""))
            tbl[i,j].set_facecolor("#E1F5EE" if val>0 else "#FCEBEB")
            tbl[i,j].set_text_props(
                color="#085041" if val>0 else "#791F1F",
                fontweight="bold")

    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, "exp4_summary_table.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path4}")

    return path1, path2, path3, path4


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(exp1, exp2, exp3):
    profiles, a_wait, f_wait, a_tput, f_tput = exp1
    densities, da_wait, df_wait, da_tput, df_tput = exp2

    path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment","variable","agent_type","value"])
        for i,p in enumerate(profiles):
            w.writerow(["exp1_profile",p.value,"agent",   a_wait[i]])
            w.writerow(["exp1_profile",p.value,"fixed",   f_wait[i]])
            w.writerow(["exp1_tput",   p.value,"agent",   a_tput[i]])
            w.writerow(["exp1_tput",   p.value,"fixed",   f_tput[i]])
        for i,d in enumerate(densities):
            w.writerow(["exp2_density",str(d),"agent",   da_wait[i]])
            w.writerow(["exp2_density",str(d),"fixed",   df_wait[i]])
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE 5 — EVALUATION")
    print(f"  {CYCLES_PER_RUN} cycles x {len(SEEDS)} seeds per condition")
    print("=" * 60)

    exp1 = experiment_1()
    exp2 = experiment_2()
    exp3 = experiment_3()

    print("\n== Generating charts ==")
    make_charts(exp1, exp2, exp3)
    export_csv(exp1, exp2, exp3)

    # Print summary
    profiles, a_wait, f_wait, a_tput, f_tput = exp1
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    overall_wait_imp = round(
        (sum(f_wait)-sum(a_wait))/sum(f_wait)*100, 1)
    overall_tput_imp = round(
        (sum(a_tput)-sum(f_tput))/sum(f_tput)*100, 1)
    print(f"  Overall avg wait improvement : {overall_wait_imp:+.1f}%")
    print(f"  Overall throughput improvement: {overall_tput_imp:+.1f}%")
    print(f"  Charts saved to: ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()