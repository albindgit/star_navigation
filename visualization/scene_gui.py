import matplotlib.pyplot as plt
import shapely
import numpy as np
import pathlib
from motion_control.pfmpc_ds import MotionController as PFMPC_ds
from motion_control.pfmpc_ds import ControlMode
from motion_control.pfmpc_obstacle_constraints import MotionController as PFMPC_obstacle_constraints
from motion_control.pfmpc_artificial_reference import MotionController as PFMPC_artificial_reference
from motion_control.pfmpc_artificial_reference import pol2pos
from starworlds.utils.misc import draw_shapely_polygon
from motion_control.soads import draw_vector_field


class SceneGUI:

    def __init__(self, robot, scene, xlim, ylim, show_axis=False,
                 robot_color='k', robot_markersize=14, robot_alpha=0.7,
                 reference_color='y', reference_alpha=1, reference_marker='*', reference_markersize=14,
                 obstacle_color='lightgrey', obstacle_edge_color='k', show_obs_name=False,
                 travelled_path_color='k', travelled_path_linestyle='-', travelled_path_linewidth=2):
        self.scene = scene
        self.robot = robot
        self.fig, self.ax = plt.subplots()
        if not show_axis:
            self.ax.set_axis_off()
        self.scene_handles, _ = self.scene.init_plot(self.ax, obstacle_color=obstacle_color, obstacle_edge_color=obstacle_edge_color, show_obs_name=show_obs_name, draw_p0=0, draw_ref=1, reference_color=reference_color, reference_alpha=reference_alpha, reference_marker=reference_marker, reference_markersize=reference_markersize)
        self.robot_handles, _ = robot.init_plot(ax=self.ax, color=robot_color, markersize=robot_markersize, alpha=robot_alpha)
        self.travelled_path_handle = self.ax.plot([], [], color=travelled_path_color, linestyle=travelled_path_linestyle, linewidth=travelled_path_linewidth, zorder=0)[0]
        self.ax.set_xlim(xlim), self.ax.set_ylim(ylim)
        # Simulation ctrl
        self.fig_open = True
        self.paused = True
        self.step_once = False
        self.draw_vector_field = False
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        # Memory
        self.travelled_path = []

    def update(self, robot_state=None, time=None):
        if time:
            self.ax.set_title("Time: {:.1f} s".format(time))

        self.scene.update_plot(self.scene_handles)
        # Obstacles
        # [oh[0].update_plot(oh[1]) for oh in zip(self.obstacles, self.obstacle_handles)]

        # Robot and goal position
        if robot_state is not None:
            self.travelled_path += list(self.robot.h(robot_state))
            self.robot.update_plot(robot_state, self.robot_handles)
            self.travelled_path_handle.set_data(self.travelled_path[::2], self.travelled_path[1::2])

    def on_close(self, event):
        self.fig_open = False

    def on_press(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'right':
            self.step_once = True
            self.paused = True
        # elif event.key == 'f':
        #     if len(self.scene.reference_path) > 1:
        #         print("Vector field is only shown for setpoint setup!")
        #         return
        #     self.paused = True
        #     self.step_once = True
        #     self.draw_vector_field = True
        #     draw_vector_field(self.scene.reference_path, controller.obstacles_star, gui.ax, workspace=controller.workspace_rho, n=200,
        #                       color='orange', zorder=-3)
        elif event.key == 'w':
            fig_name = input("Figure file name: ")
            self.save_fig(fig_name)
        else:
            print(event.key)

    def save_fig(self, name):
        figures_path = pathlib.PurePath(__file__).parents[0].joinpath("figures")
        title = self.ax.get_title()
        self.ax.set_title("")
        self.fig.savefig(str(figures_path.joinpath(name + ".png")), bbox_inches='tight')
        self.ax.set_title(title)


class StarobsGUI(SceneGUI):
    def __init__(self, robot, scene, init_robot_state, xlim, ylim, show_axis=False,
                 robot_color='k', robot_markersize=14, robot_alpha=0.7,
                 obstacle_color='lightgrey', obstacle_edge_color='k', show_obs_name=False,
                 reference_color='y', reference_alpha=1, reference_marker='*', reference_markersize=14,
                 travelled_path_color='k', travelled_path_linestyle='-', travelled_path_linewidth=2,
                 obstacles_star_color='r', obstacles_star_show_reference=True, obstacles_star_alpha=0.1,
                 show_time = True, show_timing = False):
        self.show_timing = show_timing
        self.show_time = show_time
        self.obstacle_star_handles = []
        self.obstacles_star_draw_options = {'fc': obstacles_star_color, 'show_reference': obstacles_star_show_reference,
                                            'alpha': obstacles_star_alpha, 'zorder': 0}
        # super().__init__(robot, scene, xlim, ylim, show_axis, robot_color, robot_markersize,
        #                  robot_alpha, obstacle_color, obstacle_edge_color, show_obs_name, goal_color, goal_marker,
        #                  goal_markersize, travelled_path_color, travelled_path_linestyle, travelled_path_linewidth)
        super().__init__(robot, scene, xlim, ylim, show_axis, robot_color, robot_markersize,
                         robot_alpha, reference_color, reference_alpha, reference_marker, reference_markersize,
                         obstacle_color, obstacle_edge_color, show_obs_name, travelled_path_color, travelled_path_linestyle, travelled_path_linewidth)

        super().update(init_robot_state, 0)

    def update(self, robot_state=None, time=None, controller=None, u=None):
        # SceneFig update
        super().update(robot_state)

        title = "Time: {:.1f} s".format(time) if self.show_time else ""
        # if self.show_timing:
        #     if controller is None:
        #         prev_title = self.ax.get_title()
        #         time_timing = prev_title.split('\n')
        #         title += "\n" + time_timing[-1]
        #     else:
        #         title += "\nCompute Time ({:.1f} / {:.1f} / {:.1f})".format(controller.timing['workspace'],
        #                                                                     controller.timing['target'],
        #                                                                     controller.timing['mpc'])
        self.ax.set_title(title)

        if controller is None:
            return

        # Star obstacles
        [h.remove() for h in self.obstacle_star_handles if h is not None]
        self.obstacle_star_handles = []
        if controller.obstacles_star is not None:
            for o in controller.obstacles_star:
                lh, _ = o.draw(ax=self.ax, **self.obstacles_star_draw_options)
                self.obstacle_star_handles += lh

class PFMPCGUI(SceneGUI):

    def __init__(self, robot, scene, init_robot_state, xlim, ylim, show_axis=False,
                 robot_color='c', robot_markersize=10, robot_alpha=1,
                 obstacle_color='lightgrey', obstacle_edge_color='k', show_obs_name=False,
                 reference_color='y', reference_alpha=1, reference_marker='*', reference_markersize=10,
                 travelled_path_color='k', travelled_path_linestyle='--', travelled_path_linewidth=2,
                 theta_pos_color='y', theta_pos_marker='*', theta_pos_markersize=10,
                 s1_pos_color='None', s1_pos_marker='+', s1_pos_markersize=10,
                 pg_color='b', pg_marker='*', pg_markersize=10,
                 receding_path_color='g', receding_path_linestyle='-', receding_path_linewidth=1, receding_path_marker=None,
                 mpc_path_color='k', mpc_path_linestyle='-', mpc_path_linewidth=1,
                 base_sol_color='None', base_sol_linestyle=':', base_sol_linewidth=1,
                 mpc_artificial_path_color='y', mpc_artificial_path_linestyle='-.', mpc_artificial_path_linewidth=4,
                 mpc_tunnel_color='r', mpc_tunnel_linestyle='--', mpc_tunnel_linewidth=1, mpc_tunnel_alpha=1,
                 obstacles_star_color='b', obstacles_star_show_reference=False, obstacles_star_alpha=0.2,
                 workspace_rho_color='None', workspace_rho_show_reference=False, workspace_rho_alpha=0.2,
                 workspace_horizon_color='tab:blue', workspace_horizon_linestyle='--', workspace_horizon_linewidth=1,
                 show_u_history=False, u_history_color='k', u_history_marker='o',
                 show_mpc_solution=False, show_mpc_cost=False, controller=None,
                 indicate_sbc=True, sbc_color='r',
                 show_time=True, show_timing=False):
        self.show_timing = show_timing
        self.show_time = show_time
        self.indicate_sbc = indicate_sbc
        self.robot_color, self.sbc_color = robot_color, sbc_color
        self.obstacle_star_handles = []
        self.obstacles_star_draw_options = {'fc': obstacles_star_color, 'show_reference': obstacles_star_show_reference,
                                            'alpha': obstacles_star_alpha, 'zorder': 0}
        self.workspace_rho_draw_options = {'ec': workspace_rho_color, 'fc': 'None', 'ls': '--',
                                           'show_reference': workspace_rho_show_reference,
                                           'alpha': workspace_rho_alpha, 'zorder': 0}
        super().__init__(robot, scene, xlim, ylim, show_axis, robot_color, robot_markersize,
                         robot_alpha, reference_color, reference_alpha, reference_marker, reference_markersize, obstacle_color, obstacle_edge_color, show_obs_name, travelled_path_color, travelled_path_linestyle, travelled_path_linewidth)
        self.theta_pos_handle = self.ax.plot([], [], color=theta_pos_color, marker=theta_pos_marker, markersize=theta_pos_markersize, zorder=0)[0]
        self.pg_handle = self.ax.plot([], [], color=pg_color, marker=pg_marker, markersize=pg_markersize)[0]
        self.receding_path_handle = self.ax.plot([], [], color=receding_path_color, linestyle=receding_path_linestyle, linewidth=receding_path_linewidth, marker=receding_path_marker, zorder=0)[0]
        self.mpc_path_handle = self.ax.plot([], [], color=mpc_path_color, linestyle=mpc_path_linestyle, linewidth=mpc_path_linewidth, zorder=0)[0]
        self.s1_pos_handle = self.ax.plot([], [], color=s1_pos_color, marker=s1_pos_marker, markersize=s1_pos_markersize, zorder=0)[0]
        # Tunnel and base_solution for PFMPC_DS
        self.mpc_tunnel_handle = self.ax.plot([], [], color=mpc_tunnel_color, linestyle=mpc_tunnel_linestyle, linewidth=mpc_tunnel_linewidth, alpha=mpc_tunnel_alpha, zorder=0)[0]
        self.base_sol_path_handle = self.ax.plot([], [], color=base_sol_color, linestyle=base_sol_linestyle, linewidth=base_sol_linewidth, zorder=0, dashes=(0.8, 0.8))[0]

        self.workspace_horizon_handle = self.ax.plot([], [], color=workspace_horizon_color, linestyle=workspace_horizon_linestyle, linewidth=workspace_horizon_linewidth)[0]

        # Artifical path for PFMPC_artificial
        self.mpc_artificial_path_handle = self.ax.plot([], [], color=mpc_artificial_path_color, linestyle=mpc_artificial_path_linestyle, linewidth=mpc_artificial_path_linewidth, zorder=4)[0]

        self.show_mpc_solution = show_mpc_solution
        self.show_mpc_cost = show_mpc_cost
        self.show_u_history = show_u_history
        if show_u_history:
            self.fig_u_history, self.ax_u_history = plt.subplots(self.robot.nu, 1)
            self.u_history = []
            self.u_history_handles = []
            self.u_infeasible = []
            self.u_infeasible_handles = []
            for i in range(self.robot.nu):
                self.u_history_handles += self.ax_u_history[i].plot([], [], color=u_history_color, marker=u_history_marker)
                self.u_infeasible_handles += self.ax_u_history[i].plot([], [], color='r', marker=u_history_marker, linestyle="None")
                u_span = self.robot.u_max[i] - self.robot.u_min[i]
                self.ax_u_history[i].set_ylim([self.robot.u_min[i]-0.1*u_span, self.robot.u_max[i]+0.1*u_span])
                self.ax_u_history[i].plot([0, 1e10], [robot.u_min[i], robot.u_min[i]], 'r--')
                self.ax_u_history[i].plot([0, 1e10], [robot.u_max[i], robot.u_max[i]], 'r--')

        if show_mpc_solution:
            N = controller.mpc.build_params['N']
            self.fig_mpc_solution, self.ax_mpc_solution = plt.subplots(2, 3)
            self.s_handle = self.ax_mpc_solution[0, 0].plot(np.linspace(1, N, N), [None] * N, '-o')[0]
            self.lam_rho_handle = self.ax_mpc_solution[0, 0].plot([0, N], [None, None], 'r--')[0]
            self.ax_mpc_solution[0, 0].plot([0, N], [N, N], 'r--')
            self.ax_mpc_solution[0, 0].set_ylim(0, 1.1*N)
            self.ax_mpc_solution[0, 0].set_title('Path coordinate, s')
            self.rho_handle = self.ax_mpc_solution[1, 0].plot([0, N], [None, None], 'k--')[0]
            self.emax_handle = self.ax_mpc_solution[1, 0].plot([0, N], [None, None], 'r--')[0]
            self.e_handle = self.ax_mpc_solution[1, 0].plot(np.linspace(0, N, N+1, '-o'), [None] * (N+1), '-o')[0]
            self.mpc_artificial_error_handle = self.ax_mpc_solution[1, 0].plot(np.linspace(0, N, N+1, '-o'), [None] * (N+1), 'g-o')[0]

            self.ax_mpc_solution[1, 0].set_xlim(0, N)
            if hasattr(controller, 'rho0'):
                self.ax_mpc_solution[1, 0].set_ylim(0, 1.5*controller.params['rho0'])
            self.ax_mpc_solution[1, 0].set_title('Tracking error')
            # Assumes 2 control signals
            self.u1_handle = self.ax_mpc_solution[0, 1].plot(np.linspace(-1, N-1, N+1), [None] * (N + 1), '-o')[0]
            self.ax_mpc_solution[0, 1].plot([0, N], [robot.u_min[0], robot.u_min[0]], 'r--')
            self.ax_mpc_solution[0, 1].plot([0, N], [robot.u_max[0], robot.u_max[0]], 'r--')
            self.ax_mpc_solution[0, 1].set_title('u1')
            self.u2_handle = self.ax_mpc_solution[1, 1].plot(np.linspace(-1, N-1, N+1), [None] * (N + 1), '-o')[0]
            self.ax_mpc_solution[1, 1].plot([0, N], [robot.u_min[1], robot.u_min[1]], 'r--')
            self.ax_mpc_solution[1, 1].plot([0, N], [robot.u_max[1], robot.u_max[1]], 'r--')
            self.ax_mpc_solution[1, 1].set_title('u2')
            if show_mpc_cost:
                # draw the initial pie chart
                self.cost_ax_handle = self.ax_mpc_solution[0, 2]
                self.cost_u_ax_handle = self.ax_mpc_solution[1, 2]

        plt.figure(self.fig)

        super().update(init_robot_state, 0)
        # self.update(init_robot_state, None, 0)

    def update(self, robot_state, time, controller=None, u=None):
        # SceneFig update
        super().update(robot_state)

        title = "Time: {:.1f} s".format(time) if self.show_time else ""
        if self.show_timing:
            if controller is None:
                prev_title = self.ax.get_title()
                time_timing = prev_title.split('\n')
                title += "\n" + time_timing[-1]
            else:
                title += "\nCompute Time ({:.1f} / {:.1f} / {:.1f})".format(controller.timing['workspace'],
                                                                            controller.timing['target'],
                                                                            controller.timing['mpc'])
        self.ax.set_title(title)

        if controller is None:
            return

        if isinstance(controller, PFMPC_ds):
            if controller.pg is None:
                self.pg_handle.set_data([],[])
            else:
                self.pg_handle.set_data(*controller.pg)
            # Star environment
            [h.remove() for h in self.obstacle_star_handles if h is not None]
            self.obstacle_star_handles = []
            for o in controller.obstacles_star:
                lh, _ = o.draw(ax=self.ax, **self.obstacles_star_draw_options)
                self.obstacle_star_handles += lh
            if self.scene.workspace is not None:
                # lh, _ = workspace.draw(ax=self.ax, **self.workspace_rho_draw_options)
                # self.obstacle_star_handles += lh
                # lh, _ = controller.workspace_rho.draw(ax=self.ax, fc='w', zorder=-8, show_reference=0)
                lh, _ = controller.workspace_rho.draw(ax=self.ax, **self.workspace_rho_draw_options)
                self.obstacle_star_handles += lh
            if controller.params['workspace_horizon'] > 0:
                self.workspace_horizon_handle.set_data(*shapely.geometry.Point(self.robot.h(robot_state)).buffer(controller.params['workspace_horizon']).exterior.xy)

            if self.indicate_sbc and controller.mode == ControlMode.SBC:
                self.robot_handles[0].set_color(self.sbc_color)
            else:
                self.robot_handles[0].set_color(self.robot_color)

            # SBC base solution
            if controller.sol_feasible:
                self.base_sol_path_handle.set_data([], [])
            else:
                x_sol_base, _, _ = controller.mpc.sol2state(controller.mpc.base_solution(robot_state, controller.path_pol, 0), robot_state, controller.path_pol)
                base_sol_path = np.array(
                    [self.robot.h(x_sol_base[k * self.robot.nx:(k + 1) * self.robot.nx]) for k in range(controller.params['N'] + 1)])
                self.base_sol_path_handle.set_data(base_sol_path[:, 0], base_sol_path[:, 1])

        # RHRP
        self.receding_path_handle.set_data(controller.rhrp_path[:, 0], controller.rhrp_path[:, 1])
        if controller.reference_path.geom_type == 'LineString':
            self.theta_pos_handle.set_data(*controller.reference_path.interpolate(controller.theta).coords[0])

        if controller.sol_feasible:
            if isinstance(controller, PFMPC_ds):
                u_sol, ds_sol = controller.mpc.sol2uds(controller.solution)
                x_sol, s_sol, e_sol = controller.mpc.sol2state(controller.solution, robot_state, controller.path_pol)
                # Draw tunnel
                e_max = controller.rho - controller.epsilon
                tunnel_polygon = shapely.geometry.LineString(
                    list(zip(controller.rhrp_path[:, 0], controller.rhrp_path[:, 1]))).buffer(
                    e_max)
                if tunnel_polygon.geom_type == 'Polygon':
                    self.mpc_tunnel_handle.set_data(*tunnel_polygon.exterior.xy)
                else:
                    print("[SceneFig]: Tunnel polygon not polygon.")

            if isinstance(controller, PFMPC_obstacle_constraints):
                u_sol, ds_sol = controller.mpc.sol2uds(controller.solution)
                x_sol, s_sol, e_sol = controller.mpc.sol2state(controller.solution, robot_state, controller.path_pol)

            if isinstance(controller, PFMPC_artificial_reference):
                xa0, u_sol, ua_sol, ds_sol = controller.mpc.sol2xa0uuaw(controller.solution)
                x_sol, s_sol, xa_sol = controller.mpc.sol2state(controller.solution, robot_state, controller.path_pol)
                ref_sol = np.array([pol2pos(controller.path_pol, s, controller.mpc.build_params['n_pol']) for s in s_sol])
                mpc_path = np.array([self.robot.h(x_sol[k * self.robot.nx:(k + 1) * self.robot.nx]) for k in
                                     range(controller.params['N'] + 1)])
                mpc_artificial_path = np.array([self.robot.h(xa_sol[k * self.robot.nx:(k + 1) * self.robot.nx])
                                                for k in range(controller.params['N'] + 1)])
                e_sol = np.linalg.norm(ref_sol - mpc_path, axis=1)
                self.mpc_artificial_path_handle.set_data(mpc_artificial_path[:, 0], mpc_artificial_path[:, 1])
                self.receding_path_handle.set_data(ref_sol[:, 0], ref_sol[:, 1])

            # print(x_sol)
            mpc_path = np.array([self.robot.h(x_sol[k * self.robot.nx:(k + 1) * self.robot.nx]) for k in range(controller.params['N'] + 1)])
            self.mpc_path_handle.set_data(mpc_path[:, 0], mpc_path[:, 1])
            self.s1_pos_handle.set_data(*pol2pos(controller.path_pol, s_sol[1], controller.mpc.build_params['n_pol']))


        if self.show_u_history:
            self.u_infeasible += u if not controller.sol_feasible else [None] * self.robot.nu
            self.u_history += u
            for i in range(self.robot.nu):
                if len(self.u_history) > 2:
                    N = len(self.u_history) // self.robot.nu
                    self.u_history_handles[i].set_data(np.arange(0, N), self.u_history[i::self.robot.nu])
                    self.u_infeasible_handles[i].set_data(np.arange(0, N), self.u_infeasible[i::self.robot.nu])
                    self.ax_u_history[i].set_xlim([0, N])

        # MPC solution plot
        if self.show_mpc_solution:
            sol_color = 'tab:blue' if controller.sol_feasible else 'tab:brown'
            self.s_handle.set_ydata(np.array(s_sol[1:]) / (controller.params['dt'] * controller.mpc.build_params['w_max']))
            self.s_handle.set_color(sol_color)
            self.e_handle.set_ydata(e_sol)
            self.e_handle.set_color(sol_color)
            if isinstance(controller, PFMPC_ds):
                self.lam_rho_handle.set_ydata(controller.lam_rho * np.ones(2))
                self.emax_handle.set_ydata([e_max, e_max])
                self.rho_handle.set_ydata([controller.rho, controller.rho])
            if isinstance(controller, PFMPC_artificial_reference):
                ea_sol = np.linalg.norm(mpc_path - mpc_artificial_path, axis=1)
                self.mpc_artificial_error_handle.set_ydata(ea_sol)
            self.u1_handle.set_ydata([controller.u_prev[0]] + u_sol[::2])
            self.u2_handle.set_ydata([controller.u_prev[1]] + u_sol[1::2])
            self.u1_handle.set_color(sol_color)
            self.u2_handle.set_color(sol_color)
            if self.show_mpc_cost:
                if isinstance(controller, PFMPC_ds):
                    cost = controller.mpc.sol2cost(controller.solution, robot_state, controller.path_pol, controller.params, controller.u_prev)
                    tot_cost = cost['s'] + cost['e'] + cost['u'] + cost['ud']
                    print("cost: {:.3f}.  s: {:2.0f}%, e: {:2.0f}%, u: {:2.0f}%, ud: {:2.0f}%".format(tot_cost, 100*cost['s']/tot_cost, 100*cost['e']/tot_cost, 100*cost['u']/tot_cost, 100*cost['ud']/tot_cost))

                if isinstance(controller, PFMPC_artificial_reference):

                    o_par = controller.extract_obs_par(self.obstacles).copy()
                    # Add safety margin in constraints
                    for i in range(controller.mpc.build_params['max_No_ell']):
                        o_par[6 * i + 4] += 0.2
                        o_par[6 * i + 5] += 0.2
                    print(controller.mpc.sol2cost(controller.solution, robot_state, controller.path_pol, o_par, controller.params))
                    # print(x_sol)
                    # print(xa_sol)
                    # print(u_sol)
                    # print(ua_sol)
                    # print(ds_sol)
            # if self.show_mpc_cost and cost is not None:
            #     cost_dict = dict(list(cost.items())[:-4])
            #     u_cost_dict = dict(list(cost.items())[-4:])
            #     self.cost_ax_handle.clear()
            #     self.cost_ax_handle.pie(cost_dict.values(), labels=cost_dict.keys(), wedgeprops=dict(width=0.5))
            #     self.cost_u_ax_handle.clear()
            #     self.cost_u_ax_handle.pie(u_cost_dict.values(), labels=u_cost_dict.keys(), wedgeprops=dict(width=0.5))
