#!/usr/bin/env python3
"""Plot the .npy dumps produced by opti_pessi_interface_mpc.

Reads the same config/*.info files the C++ node reads, so there is a single source of truth for
the goal, the start pose and the horizon. Run it from ~/RaNAV, where the node wrote the .npy dumps:

    plot_mpc.py <scenario>

<scenario> is 1-5 or S1-S5 (config/scenario_S<n>.info), live (config/scenario_live.info, the
controller's 4-slot OCP), or the path of any scenario .info file.
"""
import sys
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_utils as plt_ut  # noqa: E402

from info_parser import load_info  # noqa: E402


CONFIG_DIR = "src/legged_control/opti_pessi_interface/config/"


def scenario_file(name):
    """1-4 or S1-S4, live, or a path: the scenario file the node was run on."""
    if name.endswith(".info") or os.path.isfile(name):
        return name
    if name.isdigit():
        name = "S" + name
    return CONFIG_DIR + "scenario_" + name + ".info"


def obstacle_slots(scenario):
    """Obstacle slots of the OCP: obstacles.numSlots, default obstacles.numObstacles (as the C++ loader)."""
    obstacles = scenario["obstacles"]
    return int(obstacles.get("numSlots", obstacles["numObstacles"]))


def load_obstacles(path):
    y = np.load(path)
    return y[None, ...] if y.ndim == 2 else y


def animate_trot(X, p0_next, p1_next, obstacles, start, goal, dt, horizon, fig_name,
                 sub_steps=8, speed=1.0, save=True, show=False):
    """Animate the closed loop: for every CoM position show the matching p0+ / p1+ footholds.

    Between two simulation steps the CoM and the swinging foot are interpolated over `sub_steps`
    frames, so the diagonal pair that keeps its foothold reads as stance and the other one as
    swing, i.e. a trot. `speed` multiplies the playback rate.
    """
    steps = dt.size
    if steps < 2:
        return None

    # (foot, xy, step): the applied input's next foothold for each diagonal pair.
    P = np.stack([p0_next[:, :steps], p1_next[:, :steps]])
    num_obstacles = obstacles.shape[0]
    n_frames = (steps - 1) * sub_steps

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    pts_x = np.concatenate([X[0, :steps], P[:, 0, :].ravel(), obstacles[:, 0, :steps].ravel(),
                            [start[0], goal[0]]])
    pts_y = np.concatenate([X[1, :steps], P[:, 1, :].ravel(), obstacles[:, 1, :steps].ravel(),
                            [start[1], goal[1]]])
    pad = 0.15 * max(np.ptp(pts_x), np.ptp(pts_y), 1e-3)
    ax.set_xlim(pts_x.min() - pad, pts_x.max() + pad)
    ax.set_ylim(pts_y.min() - pad, pts_y.max() + pad)

    ax.plot(start[0], start[1], "sb", label="Start")
    ax.plot(goal[0], goal[1], "vb", label="Goal")

    (com_trail,) = ax.plot([], [], "-c", lw=1.5, alpha=0.6, label="Opti-Pessi MPC")
    (com_dot,) = ax.plot([], [], "oc", ms=9, mec="k", mew=0.8, zorder=5)
    (support,) = ax.plot([], [], "-", color="0.6", lw=1.0)
    (foot0,) = ax.plot([], [], "^g", ms=11, label="p0+")
    (foot1,) = ax.plot([], [], "vm", ms=11, label="p1+")
    (swing_path,) = ax.plot([], [], ":", color="0.4", lw=1.0)
    obs_dots = [ax.plot([], [], "or", ms=9, label="Obstacle" if j == 0 else None)[0]
                for j in range(num_obstacles)]
    obs_trails = [ax.plot([], [], "-r", lw=1.0, alpha=0.5)[0] for _ in range(num_obstacles)]
    ax.legend(loc="best")

    def update(frame):
        i, s = divmod(frame, sub_steps)
        t = s / sub_steps

        com = (1 - t) * X[0:2, i] + t * X[0:2, i + 1]
        com_trail.set_data(np.append(X[0, : i + 1], com[0]), np.append(X[1, : i + 1], com[1]))
        com_dot.set_data([com[0]], [com[1]])

        feet = np.empty((2, 2))
        swing = -1
        for k in range(2):
            a, b = P[k, :, i], P[k, :, i + 1]
            if np.linalg.norm(b - a) > 1e-6:
                swing = k
                feet[k] = (1 - t) * a + t * b
            else:
                feet[k] = a
        foot0.set_data([feet[0, 0]], [feet[0, 1]])
        foot1.set_data([feet[1, 0]], [feet[1, 1]])
        # Stance foot filled, swinging foot hollow, plus the segment it is travelling along.
        foot0.set_markerfacecolor("none" if swing == 0 else "g")
        foot1.set_markerfacecolor("none" if swing == 1 else "m")
        if swing >= 0:
            swing_path.set_data([P[swing, 0, i], P[swing, 0, i + 1]],
                                [P[swing, 1, i], P[swing, 1, i + 1]])
        else:
            swing_path.set_data([], [])
        support.set_data([feet[0, 0], feet[1, 0]], [feet[0, 1], feet[1, 1]])

        for j in range(num_obstacles):
            o = (1 - t) * obstacles[j, 0:2, i] + t * obstacles[j, 0:2, i + 1]
            obs_dots[j].set_data([o[0]], [o[1]])
            obs_trails[j].set_data(np.append(obstacles[j, 0, : i + 1], o[0]),
                                   np.append(obstacles[j, 1, : i + 1], o[1]))

        ax.set_title("Trot, N = %d, t = %.2f s" % (horizon, dt[: i + 1].sum() - (1 - t) * dt[i]))
        return [com_trail, com_dot, support, foot0, foot1, swing_path, *obs_dots, *obs_trails]

    # GIF viewers clamp frame delays below 20 ms, so past 50 fps the speed-up drops frames instead.
    base_fps = int(np.clip(round(sub_steps / max(dt.mean(), 1e-3)), 5, 30))
    stride = max(1, int(np.ceil(speed * base_fps / 50.0)))
    fps = speed * base_fps / stride
    frames = list(range(0, n_frames, stride))
    if frames[-1] != n_frames - 1:
        frames.append(n_frames - 1)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps, blit=False, repeat=True)

    if save:
        out = Path(plt_ut.FIGURE_PATH) / (fig_name.replace(" ", "_") + "opti_pessi_trot.gif")
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out), writer=PillowWriter(fps=fps), dpi=100)
        print("Animazione salvata: %s" % out)
    if show:
        plt.show()
    return anim


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    task = load_info(CONFIG_DIR + "task.info")
    scenario = load_info(scenario_file(argv[1]))

    horizon = int(task["horizon"]["N"])
    slots = obstacle_slots(scenario)
    start = np.array([float(task["initialState"]["(0,0)"]), float(task["initialState"]["(1,0)"])])
    goal = np.array([float(scenario["goal"]["(0,0)"]), float(scenario["goal"]["(1,0)"])])
    fig_name = scenario["simulation"]["figName"].strip('"')
    save_results = True
    save_dt = False
    clean_npy_files = True
    save_animation = True
    show_animation = False
    anim_sub_steps = 8
    anim_speed = 10.0

    suffix = fig_name + "_opti_pessi.npy"
    X = np.load("x_quad_" + suffix)
    U = np.load("u_" + suffix)
    obstacles = load_obstacles("y_obs_" + suffix)
    # Solve time of every step [s]; dumps written before 2026-10-15 have none.
    solve_times = np.load("t_solve_" + suffix).ravel() if os.path.exists("t_solve_" + suffix) else np.empty(0)

    dt = U[5, :]
    steps = dt.size

    # Next footholds, i.e. the applied input's first four rows (RobotU::P0X..P1Y).
    p0_next = U[0:2, :]
    p1_next = U[2:4, :]
    num_obstacles = obstacles.shape[0]

    # Contact-phase durations.
    plt.figure()
    plt.plot(dt, "b", label="Opti-Pessi MPC")
    plt.yticks(0.1 * np.arange(6))
    plt.locator_params(axis="x", integer=True)
    plt.xlabel("Simulation step")
    plt.ylabel("Time [s]")
    plt.grid(True)
    plt.legend()
    if save_results and save_dt:
        plt_ut.saveFigure(fig_name + "dt")

    # Closed-loop trajectory, fading in with simulation time.
    plt.figure()
    title = "OCS2, N = %d, %d obstacle slot%s, sim_T = %.2f s" % (horizon, slots, "" if slots == 1 else "s", dt.sum())
    if solve_times.size:
        # Mean without the first solve (CppAD warm-up), as the node prints it.
        title += ", solve_T = %.2f ms" % (1e3 * np.mean(solve_times[1:] if solve_times.size > 1 else solve_times))
    plt.title(title)
    fade = (1 - 0.4) / max(steps, 1)
    for i in range(steps):
        alpha = 0.2 + fade * (i + 1)
        plt.plot(X[0, i : i + 2], X[1, i : i + 2], "-c", alpha=alpha, marker=".")
        for j in range(num_obstacles):
            plt.plot(obstacles[j, 0, i : i + 2], obstacles[j, 1, i : i + 2], "-r", alpha=alpha, marker=".")
        if i == steps - 1:
            plt.plot(X[0, i : i + 2], X[1, i : i + 2], "-c", label="Opti-Pessi MPC", alpha=alpha)
            for j in range(num_obstacles):
                plt.plot(obstacles[j, 0, i : i + 2], obstacles[j, 1, i : i + 2], "-r",
                         label="Obstacle" if j == 0 else None, alpha=alpha)
    # Next footholds p0+ / p1+ and the support segment they span, fading like the CoM path.
    for i in range(steps):
        alpha = 0.2 + fade * (i + 1)
        plt.plot([p0_next[0, i], p1_next[0, i]], [p0_next[1, i], p1_next[1, i]], "-", color="0.6", alpha=alpha, lw=0.8)
        plt.plot(p0_next[0, i], p0_next[1, i], "^", color="g", alpha=alpha, ms=5)
        plt.plot(p1_next[0, i], p1_next[1, i], "v", color="m", alpha=alpha, ms=5)
    if steps:
        plt.plot(p0_next[0, -1], p0_next[1, -1], "^g", label="p0+")
        plt.plot(p1_next[0, -1], p1_next[1, -1], "vm", label="p1+")

    for j in range(num_obstacles):
        plt.plot(obstacles[j, 0, -1], obstacles[j, 1, -1], "-or")
        plt.plot(obstacles[j, 0, 0], obstacles[j, 1, 0], "sr")
    plt.plot(X[0, -1], X[1, -1], "-oc")
    plt.plot(start[0], start[1], "sb", label="Start")
    plt.plot(goal[0], goal[1], "vb", label="Goal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.legend()
    if save_results:
        plt_ut.saveFigure(fig_name + "opti_pessi")
    # plt.show()

    # Trot animation: CoM sliding along X with its p0+ / p1+ footholds at each step.
    if save_animation or show_animation:
        animate_trot(X, p0_next, p1_next, obstacles, start, goal, dt, horizon, fig_name,
                     sub_steps=anim_sub_steps, speed=anim_speed, save=save_animation, show=show_animation)


    if clean_npy_files:
        x_file = "x_quad_" + suffix
        u_file = "u_" + suffix
        y_file = "y_obs_" + suffix
        t_file = "t_solve_" + suffix
        npy_files_to_remove = [x_file, u_file, y_file, t_file]
        for f in npy_files_to_remove:
            if os.path.exists(f):
                os.remove(f)
                print(f"Rimosso file: {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
