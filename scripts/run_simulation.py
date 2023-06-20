import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from visualization import PFMPCGUI, VideoWriter, StarobsGUI
from starworlds.utils.misc import tic, toc
from motion_control.pfmpc_ds.motion_controller import ControlMode
from motion_control.soads import draw_vector_field

make_video = 0

# ctrl_param_file = 'tunnel_mpc_params.yaml'
ctrl_param_file = 'pfmpc_ds_params.yaml'
# ctrl_param_file = 'pfmpc_obstacle_constraints_params.yaml'
# ctrl_param_file = 'pfmpc_artificial_reference_params.yaml'
# ctrl_param_file = 'dmp_ctrl_params.yaml'
robot_type_id = 1
scene_id = None
# scene_id = 13
verbosity = 0
show_mpc_solution = 0
show_mpc_cost = 0
show_u_history = 0
show_timing = 0

scene, robot, controller, x0 = load_config(ctrl_param_file=ctrl_param_file, robot_type_id=robot_type_id, scene_id=scene_id, verbosity=verbosity)

# x0[:2] = [0., -5.]


if hasattr(controller, 'mpc'):
    gui = PFMPCGUI(robot, scene, x0, scene.xlim, scene.ylim,
                   show_mpc_solution=show_mpc_solution, show_mpc_cost=show_mpc_cost, show_u_history=show_u_history,
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
                   mpc_path_linestyle='-', mpc_path_linewidth=4, mpc_path_color='k',
                   mpc_tunnel_color='None',
                   travelled_path_linestyle='--', travelled_path_linewidth=3,
                   receding_path_color='g',
                   receding_path_linewidth=2,
                   show_time=1, show_timing=1, show_axis=0
                   )
else:
    gui = StarobsGUI(robot, scene, x0, scene.xlim, scene.ylim)

workspaces = scene.workspace if isinstance(scene.workspace, list) else [scene.workspace]

# Initialize
T_max = 200
theta_max = 100
dt = 0.01
dt_gui = 0.1
ctrl_sim_dt_ratio = int(controller.params['dt'] / dt)
gui_sim_dt_ratio = int(dt_gui / dt)
k = 0
x = x0
active_ws_idx = 0
converged = False
collision = False
timing = {'workspace': [], 'target': [], 'mpc': []}
if 'workspace_detailed' in controller.timing:
    workspace_timing_detailed = {'cluster': [], 'admker': [], 'hull': [], 'convex': []}

# Init video writing
if make_video:
    video_name = input("Video file name: ")
    video_writer = VideoWriter(video_name, 1/dt_gui)
    gui.paused = False
else:
    # Init gui plot
    gui.update(x, k*dt)
    plt.pause(0.005)


ls = []

# for i, attr in enumerate(scene.ws_attractors[:-1]):
#     gui.ax.plot(*attr, 'y*', markersize=10)
# gui.ax.plot(*scene.ws_attractors[-1], 'g*', markersize=14, zorder=-1)

# xlim_closeup, ylim_closeup = [-4.2, -0.8], [-5.2, -2.4]
# gui.ax.plot([xlim_closeup[0], xlim_closeup[1], xlim_closeup[1], xlim_closeup[0], xlim_closeup[0]],
#             [ylim_closeup[0], ylim_closeup[0], ylim_closeup[1], ylim_closeup[1], ylim_closeup[0]], 'k-')

while gui.fig_open and k*dt <= T_max and controller.theta <= theta_max and not converged and not collision:
    if (not gui.paused or gui.step_once):
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

            # # For Fig 9.a
            # if k == 1120:
            #     nominal_rhrp, _ =  controller.nominal_rhrp(controller.r_plus, scene.obstacles[0].polygon())

            # Compute mpc
            controller.update_policy(x, scene.obstacles, workspace=workspaces[active_ws_idx])
            u = controller.compute_u(x)
            gui.update(x, k*dt, controller, u)
            # Update timing
            tot_time = 0
            for key in timing.keys():
                timing[key] += [controller.timing[key]]
            if 'workspace_detailed' in controller.timing and controller.timing['workspace_detailed'] is not None:
                for i, key in enumerate(workspace_timing_detailed.keys()):
                    workspace_timing_detailed[key] += [controller.timing['workspace_detailed'][i]]
                # tot_time += v
            # timing['tot'] += [tot_time]

            # if controller.mode == ControlMode.SBC:
            #     print("SBC at time {:.2f} (k={:.0f})".format(k*dt, k))

        # # For Fig 8.b
        # if k == 1250:
        #     xmin, ymin, xmax, ymax = controller.workspace_rho.polygon().bounds
        #     draw_vector_field(controller.pg, controller.obstacles_star, xlim=[xmin, xmax], ylim=[ymin, ymax],
        #                       ax=gui.ax, n=40, density=0.5, linewidth=1, color='tab:olive', zorder=-1)
        #     plt.pause(0.5)
        #     while not gui.fig.waitforbuttonpress(): pass
        #
        # # For Fig 8.c
        # if k == 1360:
        #     print("SBC at time {:.2f} (k={:.0f})".format(k*dt, k))
        #     xlim, ylim = gui.ax.get_xlim(), gui.ax.get_ylim()
        #     gui.ax.set_xlim([-5.5, -2.6])
        #     gui.ax.set_ylim([-4, 2])
        #     draw_vector_field(controller.pg, controller.obstacles_star, workspace=controller.workspace_rho,
        #                       ax=gui.ax, n=80, density=1.4, linewidth=1, color='tab:olive', zorder=-1)
        #     robot.draw(x, ax=gui.ax, color='k', markersize=26, alpha=1)
        #     plt.pause(0.5)
        #     while not gui.fig.waitforbuttonpress(): pass
        #     gui.ax.set_xlim(xlim)
        #     gui.ax.set_ylim(ylim)

        # # For Fig 8.d
        # if k == 1520:
        #     xlim, ylim = gui.ax.get_xlim(), gui.ax.get_ylim()
        #     gui.ax.set_xlim([-5.5, -2.6])
        #     gui.ax.set_ylim([-4, 2])
        #     gui.mpc_path_handle.set_color('k')
        #     gui.mpc_path_handle.set_linestyle('-')
        #     robot.draw(x, ax=gui.ax, color='k', markersize=26, alpha=1)
        #     plt.pause(0.5)
        #     while not gui.fig.waitforbuttonpress(): pass
        #     gui.ax.set_xlim(xlim)
        #     gui.ax.set_ylim(ylim)
        #     gui.mpc_path_handle.set_color('None')

        # # For Fig 9.a
        # if k == 1120:
        #     gui.ax.plot(*nominal_rhrp.xy, 'm-', linewidth=3, zorder=1)
        #     for o in controller.obstacles_rho:
        #         o.draw(ax=gui.ax, fc='y', zorder=0, show_reference=0)
        #     while not gui.fig.waitforbuttonpress(): pass

        # GUI update
        if k % gui_sim_dt_ratio == 0:
            gui.step_once = False
            gui.update(x, k*dt)

            if make_video:
                video_writer.add_frame(gui.fig)
                print("[VideoWriter] wrote frame at time {:.2f}/{:.2f}".format(k * dt, T_max))
            else:
                plt.pause(0.005)

        # from motion_control.soads import compute_weights
        # import shapely
        # gammas = [obs.distance_function(p) for obs in controller.obstacles_star]
        # xrs = [obs.xr() for obs in controller.obstacles_star]
        # ds = [obs.polygon().distance(shapely.geometry.Point(p)) for obs in controller.obstacles_star]
        # if controller.workspace_rho is not None:
        #     xrs += [controller.workspace_rho.xr()]
        #     gammas += [1 / controller.workspace_rho.distance_function(p+0.01)]
        #     ds += [controller.workspace_rho.polygon().exterior.distance(shapely.geometry.Point(p))]
        # ws = compute_weights(gammas)
        # ws = compute_weights(ds, gamma_lowerlimit=0)
        # if ls:
        #     [l.remove() for l in ls if l is not None]
        # ls = []
        # for xr, w, gamma, d in zip(xrs, ws, gammas, ds):
        #     ls += [gui.ax.text(*xr, "{:.2f}".format(w))]

        # while not gui.fig.waitforbuttonpress(): pass
        # gui.ax.set_xlim(np.array(xlim_closeup) + np.array([-0.01, 0.01]))
        # gui.ax.set_ylim(np.array(ylim_closeup) + np.array([-0.01, 0.01]))


        # Robot integration
        u = controller.compute_u(x)
        x, _ = robot.move(x, u, dt)
        k += 1

        # Convergence and collision check
        converged = controller.theta >= controller.theta_g and np.linalg.norm(robot.h(x)-scene.reference_path[-1]) < controller.params['convergence_tolerance']
        collision = any([o.interior_point(robot.h(x)) for o in scene.obstacles])
        if collision:
            print("Collision")
    # elif make_video:
    #     converged = 1

    if gui.paused and not gui.step_once:
        gui.fig.waitforbuttonpress()

if make_video:
    # close video writer
    video_writer.release()
    print("Finished")
    fig_open = False
else:
    gui.update(x, k*dt, controller, u)
    gui.ax.set_title("Time: {:.1f} s. Finished".format(k*dt))

    if show_timing:
        fig, ax = plt.subplots()
        ax.plot(timing['workspace'])
        ax.plot(timing['target'])
        ax.plot(timing['mpc'])
        ax.legend(['Workspace modification', 'Target path generation', 'MPC'])
        columns = ['min', 'max', 'mean']

        rows = []
        data = []
        for k, v in timing.items():
            rows += [k]
            data += [[np.min(v), np.max(v), np.mean(v)]]
        if 'workspace_detailed' in controller.timing:
            rows += [""]
            data += [[None, None, None]]
            for k, v in workspace_timing_detailed.items():
                rows += [k]
                data += [[np.min(v), np.max(v), np.mean(v)]]

        # Plot bars and create text labels for the table
        cell_text = []
        for row in range(len(rows)):
            if row == 3:
                cell_text.append(["", "", ""])
            else:
                cell_text.append(["{:.2f}".format(t) for t in data[row]])
        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              colLabels=columns,
                              loc='bottom')

        # Adjust layout to make room for the table:
        # plt.subplots_adjust(left=0.2, bottom=0.4)
        fig.tight_layout()
        # plt.subplots_adjust(hspace=0.2)

        # plt.ylabel(f"Loss in ${value_increment}'s")
        # plt.yticks(values * value_increment, ['%d' % val for val in values])
        plt.xticks([])
        plt.title('Computation time')
        #
        # fig, ax = plt.subplots()
        # ax.plot(timing['workspace'])
        # ax.plot(timing['target'])
        # ax.plot(timing['mpc'])
        # # ax.plot(timing['tot'])
        # ax.legend(['Workspace modification', 'Target path generation', 'MPC'])

    # Wait until figure closed when converged
    plt.show()
