import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from visualization import PFMPCGUI

make_video = 0

ctrl_param_file = 'pfmpc_ds_params.yaml'
robot_type_id = 3
scene_id = 13
verbosity = 0

scene, robot, controller, x0 = load_config(ctrl_param_file=ctrl_param_file, robot_type_id=robot_type_id, scene_id=scene_id, verbosity=verbosity)

workspaces = scene.workspace if isinstance(scene.workspace, list) else [scene.workspace]


# Settings
T_max = 200
theta_max = 100
dt = 0.01
ctrl_sim_dt_ratio = int(controller.params['dt'] / dt)
show_gui = True


if show_gui:
    gui = PFMPCGUI(robot, scene, x0, scene.xlim, scene.ylim,
                       controller=controller, robot_alpha=1., robot_color='k',
                       obstacles_star_alpha=0.2, obstacles_star_show_reference=0,
                       obstacles_star_color='b',
                       # workspace_rho_color='None',
                       # show_obs_name=1,
                       reference_color='y',
                       reference_markersize=10,
                       pg_markersize=10, pg_color='b', pg_marker='*',
                       theta_pos_color='g', theta_pos_marker='*', theta_pos_markersize=10,
                       s1_pos_color='g', s1_pos_marker='+', s1_pos_markersize=0,
                       mpc_path_linestyle='-', mpc_path_linewidth=4, mpc_path_color='None',
                       mpc_tunnel_color='r',
                       travelled_path_linestyle='--', travelled_path_linewidth=3,
                       receding_path_color='g',
                       receding_path_linewidth=2,
                       show_time=1, show_timing=1, show_axis=0
                       )


fig, ax = plt.subplots()
linestyles = ['-', '--', ':', '-.']
rows = []
data = []

x0_0s = [-4, -2, 0, 2]
for i, x0_0 in enumerate(x0_0s):
    x0[:2] = [x0_0, -5.]



    # Initialize
    k = 0
    x = x0
    active_ws_idx = 0
    converged = False
    collision = False
    timing = {'workspace': [], 'target': [], 'mpc': []}
    controller.reset()


    while k*dt <= T_max and controller.theta <= theta_max and not converged and not collision:
        p = robot.h(x)
        if active_ws_idx + 1 < len(workspaces):
            ws_smaller = workspaces[active_ws_idx + 1].dilated_obstacle(-0.1, id='temp')
            if controller.rhrp_path is not None and all([ws_smaller.interior_point(r) for r in controller.rhrp_path]):
                active_ws_idx += 1
                if len(scene.reference_path) == 1:
                    controller.set_reference_path([scene.ws_attractors[active_ws_idx]])

        # Move obstacles
        scene.step(dt, p)

        # Control update
        if k % ctrl_sim_dt_ratio == 0:
            if k*dt % 2 == 0:
                print(k*dt)
            # Compute mpc
            controller.update_policy(x, scene.obstacles, workspace=workspaces[active_ws_idx])
            # Update timing
            close_to_convergence = (np.linalg.norm(controller.r_plus - controller.pg) < controller.rho) and (controller.theta >= controller.theta_g)
            if not close_to_convergence:
                for key in timing.keys():
                    timing[key] += [controller.timing[key]]

            if show_gui:
                gui.update(x, k*dt, controller)
                plt.pause(0.005)

        # Robot integration
        u = controller.compute_u(x)
        x, _ = robot.move(x, u, dt)
        k += 1

        # Convergence and collision check
        converged = controller.theta >= controller.theta_g and np.linalg.norm(robot.h(x)-scene.reference_path[-1]) < controller.params['convergence_tolerance']
        collision = any([o.interior_point(robot.h(x)) for o in scene.obstacles])
        if collision:
            print("Collision")


    ax.plot(timing['workspace'],'b', ls=linestyles[i])
    ax.plot(timing['target'],'r', ls=linestyles[i])
    ax.plot(timing['mpc'],'g', ls=linestyles[i])
    ax.legend(['Workspace modification', 'Target path generation', 'MPC'])

    for k, v in timing.items():
        rows += [k]
        data += [[np.min(v), np.max(v), np.mean(v)]]
    rows += [""]

# Plot bars and create text labels for the table
columns = ['min', 'max', 'mean']
cell_text = []
for row in range(len(rows)):
    if row % 2 == 0:
        cell_text.append(["", "", ""])
    cell_text.append(["{:.2f}".format(t) for t in data[row]])
# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='bottom')

fig.tight_layout()
plt.xticks([])
plt.title('Computation time')
plt.show()

