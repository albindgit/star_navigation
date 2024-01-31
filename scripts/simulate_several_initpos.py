import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from visualization import PFMPCGUI, VideoWriter
from motion_control.pfmpc_ds import pol2pos
import shapely

# ------------------------------- #

make_video = 0

# Scenario settings
ctrl_param_file = 'pfmpc_ds_params.yaml'
robot_type_id = 1
scene_id = 22
verbosity = 0

# Simulation settings
T_max = 200
dt = 0.01
dt_gui = 0.1

# GUI settings
gui_robot_idx = 3  # GUI environment is shown for robot corresponding to this x0 idx
show_mpc_tunnel = 0  # 0: Don't show.  1: Show for gui robot.  2: Show for all robots.
show_mpc_sol = 0  # 0: Don't show.  1: Show for gui robot.  2: Show for all robots.
show_travelled_path = 2  # 0: Don't show.  1: Show for gui robot.  2: Show for all robots.

gui_robot_color, robot_color = 'tab:orange', 'k'
gui_robot_color, robot_color = 'k', 'k'
travelled_path_color, travelled_path_linestyle, travelled_path_linewidth = 'k', '--', 2
mpc_sol_color, mpc_sol_linestyle, mpc_sol_linewidth = 'tab:orange', '-', 1
mpc_tunnel_color, mpc_tunnel_linestyle, mpc_tunnel_linewidth = 'r', '--', 1
rhrp_color, rhrp_linestyle, rhrp_linewidth = 'g', '-', 2

# ------------------------------- #

# dict {scene_id: x0s}
scene_x0_mapping = {8: [[6, 0, np.pi/2], [3, 0, np.pi/2], [8, 0, np.pi/2], [4, -2, np.pi/2], [8, 2, np.pi/2]],
                    10: [[-2.5, 1.5, 0.], [2.4, -5.6, 0.], [3, .5, 0.], [2, 2.5, 0.], [-5, 1.5, 0.], [-2.8, -3.8, 0.]],
                    12: [[-1, 5, 0], [-4, -5.7, 0], [7, 0.25, 0], [1.2, 1.2, 0], [-3, 3, 0], [3.5, 3.5, 0]],
                    13: [[0, -5, np.pi/2], [-2, -5, np.pi/2], [2, -5, np.pi/2], [-4, -5, np.pi/2], [4, -5, np.pi/2]],
                    22: [[0, -5, np.pi/2], [-2, -5, np.pi/2], [2, -5, np.pi/2], [-4, -5, np.pi/2], [4, -5, np.pi/2]]}
x0s = np.array(scene_x0_mapping[scene_id])

if robot_type_id == 0:
    x0s = x0s[:, :2]

N = x0s.shape[0]

travelled_path_handles = [[]] * N
robot_handles = [[]] * N
controllers = [[]] * N
scenes = [[]] * N

assert gui_robot_idx < N
scenes[gui_robot_idx], robot, controllers[gui_robot_idx], _ = load_config(ctrl_param_file=ctrl_param_file, robot_type_id=robot_type_id, scene_id=scene_id, verbosity=verbosity)
if scene_id == 8:  # Special fix to show all initial positions in this scenario
    scenes[gui_robot_idx].xlim[1] += 2
if show_mpc_tunnel == 0:
    mpc_tunnel_color, rhrp_color = 'None', 'None'
if show_mpc_sol == 0:
    mpc_sol_color = 'None'
if show_travelled_path == 0:
    travelled_path_color = 'None'
gui = PFMPCGUI(robot, scenes[gui_robot_idx], x0s[gui_robot_idx], scenes[gui_robot_idx].xlim, scenes[gui_robot_idx].ylim,
               controller=controllers[gui_robot_idx], robot_alpha=1., robot_color=gui_robot_color, robot_markersize=10,
               reference_color='y', reference_marker='*', reference_markersize=10,
               pg_color='y', pg_markersize=10, pg_marker='*',
               theta_pos_color='c', theta_pos_marker='o', theta_pos_markersize=2,
               obstacles_star_alpha=0.2, obstacles_star_show_reference=0, obstacles_star_color='b',
               workspace_rho_color='b', workspace_rho_alpha=0.2, indicate_sbc=0,
               s1_pos_color='None',
               mpc_path_color=mpc_sol_color, mpc_path_linestyle=mpc_sol_linestyle, mpc_path_linewidth=mpc_sol_linewidth,
               mpc_tunnel_color=mpc_tunnel_color, mpc_tunnel_linestyle=mpc_tunnel_linestyle, mpc_tunnel_linewidth=mpc_tunnel_linewidth,
               travelled_path_color=travelled_path_color, travelled_path_linestyle=travelled_path_linestyle, travelled_path_linewidth=travelled_path_linewidth,
               receding_path_color=rhrp_color, receding_path_linestyle=rhrp_linestyle, receding_path_linewidth=rhrp_linewidth,
               show_time=1, show_timing=0, show_axis=0
               )

for i in range(N):
    if i == gui_robot_idx:
        continue
    scenes[i], _, controllers[i], _ = load_config(ctrl_param_file=ctrl_param_file, robot_type_id=robot_type_id, scene_id=scene_id, verbosity=verbosity)
    robot_handles[i], _ = robot.draw(x0s[i, :], ax=gui.ax, markersize=10, color=robot_color, alpha=1)
    travelled_path_handles[i] = gui.ax.plot([], [], c=travelled_path_color, ls=travelled_path_linestyle, lw=travelled_path_linewidth, zorder=0)[0]

# Initialize simulation
ctrl_sim_dt_ratio = int(controllers[0].params['dt'] / dt)
gui_sim_dt_ratio = int(dt_gui / dt)
k = 0
tmp_hs = []
travelled_path = x0s[:, :2]
xs = x0s.copy()
converged = [False] * N
collision = False

# Init video writing
if make_video:
    video_name = input("Video file name: ")
    video_writer = VideoWriter(video_name, 1/dt_gui)
    gui.paused = False
    frame_cntr = 0
else:
    # Init gui plot
    gui.update(xs[gui_robot_idx, :], k*dt)
    plt.pause(0.005)


while gui.fig_open and k*dt <= T_max and not all(converged) and not collision:
    if gui.paused and not gui.step_once:
        gui.fig.waitforbuttonpress()

    # Move obstacles
    for i in range(N):
        scenes[i].step(dt, robot.h(xs[i, :]))

    # Control update
    if k % ctrl_sim_dt_ratio == 0:
        gui.step_once = False
        for i in range(N):
            controllers[i].update_policy(xs[i, :], scenes[i].obstacles, workspace=scenes[i].workspace)


    # GUI update
    if k % gui_sim_dt_ratio == 0:
        [h.remove() for h in tmp_hs if not None]
        tmp_hs = []
        if k % ctrl_sim_dt_ratio == 0:
            gui.update(xs[gui_robot_idx, :], k*dt, controllers[gui_robot_idx], controllers[gui_robot_idx].compute_u(xs[gui_robot_idx, :]))
        else:
            gui.update(xs[gui_robot_idx, :], k*dt)
        for i in range(N):
            if i == gui_robot_idx:
                continue
            robot.update_plot(xs[i, :], robot_handles[i])
            travelled_path_handles[i].set_data(travelled_path[i, ::2], travelled_path[i, 1::2])
        if show_mpc_tunnel == 2 or show_mpc_sol == 2:
            for i in range(N):
                if i == gui_robot_idx:
                    continue
                c = controllers[i]
                x_sol, s_sol, e_sol = c.mpc.sol2state(c.solution, xs[i, :], c.path_pol)
                if show_mpc_tunnel == 2:
                    e_max = c.rho - c.epsilon
                    rhrp = np.array([pol2pos(c.path_pol, s, c.mpc.build_params['n_pol']) for s in s_sol])
                    tunnel_polygon = shapely.geometry.LineString(list(zip(rhrp[:, 0], rhrp[:, 1]))).buffer(e_max)
                    tmp_hs += gui.ax.plot(rhrp[:, 0], rhrp[:, 1], c=rhrp_color, ls=rhrp_linestyle, lw=rhrp_linewidth)
                    tmp_hs += gui.ax.plot(*tunnel_polygon.exterior.xy, c=mpc_tunnel_color, ls=mpc_tunnel_linestyle, lw=mpc_tunnel_linewidth)
                if show_mpc_sol == 2:
                    mpc_path = np.array([robot.h(x_sol[k * robot.nx:(k + 1) * robot.nx]) for k in range(c.params['N'] + 1)])
                    tmp_hs += gui.ax.plot(*mpc_path.T, c=mpc_sol_color, ls=mpc_sol_linestyle, lw=mpc_sol_linewidth)

        if make_video:
            video_writer.add_frame(gui.fig, frame_cntr)
            frame_cntr += 1
            print("[VideoWriter] wrote frame at time {:.2f}/{:.2f}".format(k * dt, T_max))
        else:
            plt.pause(0.005)

    # Robot integration
    for i in range(N):
        if converged[i]:
            u = np.zeros(robot.nu)
        else:
            u = controllers[i].compute_u(xs[i, :])
        xs[i, :], _ = robot.move(xs[i, :], u, dt)
    travelled_path = np.hstack((travelled_path, xs[:, :2]))

    # Convergence and collision check
    for i in range(N):
        converged[i] = controllers[i].theta >= controllers[i].theta_g and np.linalg.norm(robot.h(xs[i, :])-scenes[i].reference_path[-1]) < controllers[i].params['convergence_tolerance']
        if any([o.interior_point(robot.h(xs[i, :])) for o in scenes[i].obstacles]):
            collision = True
            print("Collision")

    k += 1


if make_video:
    # close video writer
    video_writer.release()
    print("Finished")
    fig_open = False
else:
    gui.update(xs[gui_robot_idx, :], k*dt)
    gui.ax.set_title("Time: {:.1f} s. Finished".format(k*dt))

    # Wait until figure closed when converged
    plt.show()
