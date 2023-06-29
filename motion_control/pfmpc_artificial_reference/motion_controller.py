from motion_control.pfmpc_artificial_reference import Mpc, pol2pos
from starworlds.utils.misc import tic, toc
import numpy as np
import shapely


class MotionController:
    def __init__(self, params, robot, reference_path, verbosity=0):
        self.params = params
        self.robot = robot
        self.mpc = Mpc(params, robot)
        self.verbosity = verbosity
        self.reference_path = shapely.geometry.Point(reference_path[0]) if len(reference_path) == 1 else shapely.geometry.LineString(reference_path)  # Waypoints
        self.theta_g = self.reference_path.length
        self.rhrp_L = self.params['N'] * self.params['dt'] * self.mpc.build_params['w_max']
        self.rhrp_s = np.linspace(0, self.rhrp_L, params['rhrp_steps'])
        self.reset()

    def reset(self):
        self.mpc.reset()
        self.theta = 0
        self.rhrp_path = None
        # self.rhrp_L = self.params['N'] * self.mpc.build_params['dp_max']
        # self.rhrp_s = np.linspace(0, self.rhrp_L, params['rhrp_steps'])
        # self.receding_ds = self.params['normalized_reference_stepsize'] * self.mpc.build_params['dp_max']
        # self.rhrp_s = np.arange(0, self.rhrp_L + self.receding_ds, self.receding_ds)
        self.path_pol = None
        self.epsilon = None
        self.solution = None
        self.sol_feasible = None
        self.u_prev = [0] * self.robot.nu
        self.timing = {'workspace': 0, 'target': 0, 'mpc': 0}

    def extract_obs_par(self, obstacles):
        n_ell_par = 3 * self.mpc.build_params['max_No_ell'] * (1 + self.mpc.build_params['N_obs_predict'])
        n_pol_par = self.mpc.build_params['max_No_pol'] * self.mpc.build_params['max_No_vert'] * 2
        obs_par = [0] * (n_ell_par + n_pol_par)
        obs_par[1:n_ell_par:3 * (1 + self.mpc.build_params['N_obs_predict'])] = [1] * self.mpc.build_params['max_No_ell']
        obs_par[2:n_ell_par:3 * (1 + self.mpc.build_params['N_obs_predict'])] = [1] * self.mpc.build_params['max_No_ell']
        obs_par_padded = obs_par.copy()
        n_ell, n_pol = 0, 0

        for o in obstacles:
            if hasattr(o, "_a"):
                # print(o.pos())
                j = n_ell * 3 * (1 + self.mpc.build_params['N_obs_predict'])
                obs_par[j] = 1  # Include obstacle
                obs_par[j + 1:j + 3] = o._a # Ellipse axes
                # Ugly coding for prediction
                mm = o._motion_model
                pos, rot, t = mm.pos().copy(), mm.rot(), mm._t
                if hasattr(mm, '_wp_idx'):
                    wp_idx = mm._wp_idx
                for k in range(self.mpc.build_params['N_obs_predict']):
                    obs_par[j + 3 + 3 * k:j + 5 + 3 * k] = mm.pos()  # Ellipse position
                    obs_par[j + 5 + 3 * k] = mm.rot()  # Ellipse orientation
                    mm.move(None, self.mpc.build_params['dt'])
                mm.set_pos(pos), mm.set_rot(rot)
                mm._t = t
                if hasattr(mm, '_wp_idx'):
                    mm._wp_idx = wp_idx

                obs_par_padded[j:j+3*(self.mpc.build_params['N_obs_predict']+1)] = obs_par[j:j+3*(self.mpc.build_params['N_obs_predict']+1)]
                obs_par_padded[j + 1:j + 3] = o._a + self.params['obstacle_padding']

                n_ell += 1

            if hasattr(o, "vertices"):
                idx = n_ell_par + n_pol * self.mpc.build_params['max_No_vert'] * 2
                vertices = shapely.ops.orient(o.polygon()).exterior.coords[:-1]
                for i in range(self.mpc.build_params['max_No_vert']):
                    obs_par[idx+i*2:idx+(i+1)*2] = vertices[i % len(vertices)]
                vertices = shapely.ops.orient(o.polygon().buffer(self.params['obstacle_padding'], quad_segs=1, cap_style=3, join_style=2)).exterior.coords[:-1]
                for i in range(self.mpc.build_params['max_No_vert']):
                    obs_par_padded[idx+i*2:idx+(i+1)*2] = vertices[i % len(vertices)]
                n_pol += 1

        # for i in range(self.mpc.build_params['max_No_ell']):
        #     j = i * 3 * (1 + self.mpc.build_params['N_obs_predict'])
        #     include_obs = obs_par[j]
        #     ell_axs = obs_par[j + 1:j + 3]
        #     ell_pos = obs_par[j + 3 + 3 * k:j + 5 + 3 * k]
        #     ell_rot = obs_par[j + 5 + 3 * k]
        #     print(include_obs,ell_axs,ell_pos,ell_rot)
        #
        # for i in range(self.mpc.build_params['max_No_pol']):
        #     j = n_ell_par + i * self.mpc.build_params['max_No_vert'] * 2
        #     print(obs_par[j : j + 2 * self.mpc.build_params['max_No_vert']])
        return obs_par, obs_par_padded

    def compute_u(self, x):
        return self.u_prev

    def update_policy(self, x, obstacles, workspace=None):
        p = self.robot.h(x)

        # Extract receding path from global target path
        t0 = tic()
        rhrp_path_sh = shapely.ops.substring(self.reference_path, start_dist=self.theta, end_dist=self.theta + self.rhrp_L)
        if rhrp_path_sh.length > 0:
            self.rhrp_path = np.array([rhrp_path_sh.interpolate(s).coords[0] for s in self.rhrp_s])
        else:
            self.rhrp_path = np.tile(rhrp_path_sh.coords[0], (len(self.rhrp_s), 1))

        # Approximate target with polynomials
        self.path_pol = np.polyfit(self.rhrp_s, self.rhrp_path[:, 0], self.params['n_pol']).tolist() + \
                        np.polyfit(self.rhrp_s, self.rhrp_path[:, 1], self.params['n_pol']).tolist()
        # Force init position to be correct
        self.path_pol[self.params['n_pol']] = self.rhrp_path[0, 0]
        self.path_pol[-1] = self.rhrp_path[0, 1]
        # Compute polyfit approximation error
        self.epsilon = max(np.linalg.norm(self.rhrp_path - np.array([pol2pos(self.path_pol, s, self.mpc.build_params['n_pol']) for s in self.rhrp_s]), axis=1))
        self.timing['target'] = toc(t0)

        t0 = tic()
        # Extract obstacle parameters (Assumes all ellipses)
        obs_par, obs_par_padded = self.extract_obs_par(obstacles)
        # Compute MPC solution
        if self.solution is not None:
            init_guess = self.solution.copy()
            init_guess[:3] = x
        else:
            init_guess = None
        solution_data = self.mpc.run(x.tolist(), self.path_pol, self.params, obs_par_padded, init_guess)
        # solution_data = self.mpc.run(x.tolist(), self.u_prev, self.path_pol, self.params, 1, 0.1, self.solution)
        if solution_data is None:
            self.sol_feasible, self.mpc_exit_status = False, "None"
        else:
            self.solution, self.mpc_exit_status = solution_data.solution, solution_data.exit_status
            self.sol_feasible = self.mpc.is_feasible(self.solution, x.tolist(), self.path_pol, obs_par, self.params, d=self.verbosity > 0)
        # self.sol_feasible = self.mpc.is_feasible(self.solution, x.tolist(), self.path_pol, 1, 0.1, d=self.verbosity > 0)

        if self.sol_feasible:
            xa0, u, ua, w = self.mpc.sol2xa0uuaw(self.solution)
            self.u_prev = u[:self.robot.nu]
            p_ref_dist = rhrp_path_sh.distance(shapely.geometry.Point(p))
            if self.theta_g > 0 and p_ref_dist < 1:
                self.theta = min(self.theta + w[0] * self.params['dt'], self.theta_g)
        else:
            self.u_prev = [0] * self.robot.nu

        self.timing['mpc'] = toc(t0)
