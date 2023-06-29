import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from visualization import PFMPCGUI, VideoWriter
from motion_control.pfmpc_ds import pol2pos
import shapely

# ------------------------------- #

make_video = 1

# Scenario settings
ctrl_param_files = ['pfmpc_ds_params.yaml','pfmpc_obstacle_constraints_params.yaml', 'pfmpc_artificial_reference_params.yaml']
ctrl_param_files = ['pfmpc_ds_params.yaml']
robot_type_id = 3
scene_id = 15
verbosity = 0

# Simulation settings
T_max = 200
dt = 0.01
dt_gui = 0.1

# GUI settings
ctrl_colors = ['k', 'r', 'y']
gui_robot_idx = 0  # GUI environment is shown for robot corresponding to this x0 idx

# ------------------------------- #


N = len(ctrl_param_files)
travelled_path_handles = [None] * N
robot_handles = [[]] * N
controllers = [[]] * N
scenes = [[]] * N

assert gui_robot_idx < N
scenes[gui_robot_idx], robot, controllers[gui_robot_idx], x0 = load_config(ctrl_param_file=ctrl_param_files[gui_robot_idx], robot_type_id=robot_type_id, scene_id=scene_id, verbosity=verbosity)
gui = PFMPCGUI(robot, scenes[gui_robot_idx], x0, scenes[gui_robot_idx].xlim, scenes[gui_robot_idx].ylim,
               controller=controllers[gui_robot_idx], robot_color=ctrl_colors[gui_robot_idx],
               reference_color='y',
               theta_pos_color='None',
               # obstacles_star_color='None',
               # pg_color='None',
               workspace_rho_color='tab:blue',
               # indicate_sbc=False,
               mpc_artificial_path_color='None',
               s1_pos_color='None',
               # mpc_path_color='None',
               # mpc_tunnel_color='None',
               # receding_path_color='None',
               travelled_path_color=ctrl_colors[gui_robot_idx]
               )

for i in range(N):
    if i == gui_robot_idx:
        continue
    scenes[i], _, controllers[i], _ = load_config(ctrl_param_file=ctrl_param_files[i], robot_type_id=3, scene_id=scene_id, verbosity=verbosity)
    robot_handles[i], _ = robot.draw(x0, ax=gui.ax, markersize=10, color=ctrl_colors[i], alpha=1)
    travelled_path_handles[i] = gui.ax.plot([], [], c=ctrl_colors[i], ls=travelled_path_linestyle, lw=travelled_path_linewidth, zorder=0)[0]

# Initialize simulation
ctrl_sim_dt_ratio = int(controllers[0].params['dt'] / dt)
gui_sim_dt_ratio = int(dt_gui / dt)
k = 0
tmp_hs = []
xs = np.tile(x0, (N, 1))
travelled_path = xs[:, :2]
converged = [False] * N
collision = False
workspaces = scenes[0].workspace if isinstance(scenes[0].workspace, list) else [scenes[0].workspace]
active_ws_idcs = [0] * N

# Init video writing
if make_video:
    video_name = input("Video file name: ")
    video_writer = VideoWriter(video_name, 1/dt_gui)
    gui.paused = False
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
            # Check current workspace
            if active_ws_idcs[i] + 1 < len(workspaces):
                ws_smaller = workspaces[active_ws_idcs[i] + 1].dilated_obstacle(-0.1, id='temp')
                if controllers[i].rhrp_path is not None and all(
                        [ws_smaller.interior_point(r) for r in controllers[i].rhrp_path]):
                    active_ws_idcs[i] += 1
                    if len(scenes[i].reference_path) == 1:
                        controllers[i].set_reference_path([scenes[i].ws_attractors[active_ws_idcs[i]]])
            # Update control policy
            controllers[i].update_policy(xs[i, :], scenes[i].obstacles, workspace=workspaces[active_ws_idcs[i]])

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

        if make_video:
            video_writer.add_frame(gui.fig)
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
