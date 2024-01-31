import matplotlib.pyplot as plt

from motion_control.pfmpc_ds import workspace_modification, path_generator, Mpc, pol2pos, rho_environment, extract_r0
from starworlds.utils.misc import tic, toc
from starworlds.obstacles import Ellipse, StarshapedPolygon, motion_model
import numpy as np
import shapely
import warnings
from enum import Enum

class ControlMode(Enum):
    SBC = 1
    MPC = 2

    class InvalidModeError(Exception):
        pass


class MotionController:
    # reference_path: list of waypoints (i.e. list of lists)
    def __init__(self, params, robot, reference_path, verbosity=0):
        self.params = params
        self.robot = robot
        self.mpc = Mpc(params, robot)
        self.rhrp_L = self.params['N'] * self.params['dt'] * self.mpc.build_params['w_max']
        self.rhrp_s = np.linspace(0, self.rhrp_L, params['rhrp_steps'])
        self.set_reference_path(reference_path)
        self.verbosity = verbosity
        # print(self.params['cs'] * self.mpc.build_params['w_max'], self.params['ce'] * self.params['rho0'])
        self.reset()

    def reset(self):
        self.mpc.reset()
        self.mode = ControlMode.SBC
        self.obstacle_clusters = None
        self.obstacles_rho = []
        self.workspace_rho = None
        self.obstacles_star = []
        self.r_plus = None
        self.theta = 0
        self.ds_generated_path = None
        self.rg = None
        self.pg_buffer = None
        self.pg = None
        self.rhrp_path = None
        self.path_pol = None
        self.rho = None
        self.epsilon = None
        self.solution = None
        self.sol_feasible = None
        self.mpc_exit_status = None
        if self.params['lambda'] * self.params['rho0'] > self.mpc.build_params['w_max'] * self.params['dt']:
            print("Poor lamda, rho0 selection. lambda*rho ({:.2f}) > w_max*dt ({:.2f})".format(self.params['lambda'] * self.params['rho0'], self.mpc.build_params['w_max'] * self.params['dt']))
        # else:
        #     print("Ok lamda, rho0 selection. {:.2f} <= {:.2f}".format(self.params['lambda'] * self.params['rho0'],
        #                                                               self.mpc.build_params['dp_max']))
        self.u_prev = [0] * self.robot.nu
        self.timing = {'workspace': 0, 'target': 0, 'mpc': 0, 'workspace_detailed': [0] * 4}

    def set_reference_path(self, reference_path):
        self.reference_path = shapely.geometry.Point(reference_path[0]) if len(reference_path) == 1 else shapely.geometry.LineString(reference_path)
        self.theta_g = self.reference_path.length

    # Extract path rs -> reference_path(theta, theta+L)
    def nominal_rhrp(self, r0, free_rho_sh):
        # Find receding global reference
        # nominal_rhrp_0 = np.array(self.reference_path.interpolate(self.theta).coords[0])
        nominal_rhrp_sh = shapely.ops.substring(self.reference_path, start_dist=self.theta, end_dist=self.theta + self.params['nominal_rhrp_horizon'])
        init_path = shapely.geometry.LineString([r0, nominal_rhrp_sh.coords[0]])
        if nominal_rhrp_sh.geom_type == 'Point':
            nominal_rhrp_sh = init_path
        else:
            nominal_rhrp_sh = shapely.ops.linemerge([init_path, nominal_rhrp_sh])
        nominal_rhrp_sh = shapely.ops.substring(nominal_rhrp_sh, start_dist=0, end_dist=self.params['nominal_rhrp_horizon'])
        path_collision_check = nominal_rhrp_sh if nominal_rhrp_sh.length > init_path.length else init_path
        collision_free = path_collision_check.within(free_rho_sh)
        return nominal_rhrp_sh, collision_free

    def ref_kappa(self, p, r0, r_lambda_rho, rho):
        if np.linalg.norm(p-r_lambda_rho) < rho:
            return r_lambda_rho
        # Intersection point of B(r0,rho) and B(rlr,rho)
        rc = 0.5 * (r0 + r_lambda_rho)
        nc = np.array([rc[1] - r0[1], r0[0] - rc[0]])
        nc /= np.linalg.norm(nc)
        d = np.sqrt(rho ** 2 - np.linalg.norm(rc - r0) ** 2)
        i1 = rc + nc * d
        i2 = rc - nc * d
        # Line coefficients for l[r0, r_lambda_rho]
        a = r0[1] - r_lambda_rho[1]
        b = r_lambda_rho[0] - r0[0]
        c = - (r0[1] * b + r0[0] * a)
        # Line coefficients for li
        i_scale = 0.01
        ai = i1[0] - p[0]
        bi = i1[1] - p[1]
        ci = 0.5 * (p.dot(p) - i1.dot(i1) + i_scale * (p.dot(i2) - p.dot(i1) - i1.dot(i2) + i1.dot(i1)))
        # Return intersection of lines
        return np.array([(b * ci - bi * c) / (a * bi - ai * b), (c * ai - ci * a) / (a * bi - ai * b)])

    def unicycle_sbc(self, x, p_ref):
        k1, k2 = 2, 2
        # k1, k2 = 1.25, 0.3
        e = np.array(x[:2]) - np.array(p_ref)
        th = x[2]

        r_th = np.arctan2(e[1], e[0]) + np.pi
        # E Theta in (-pi, pi]
        e_th = r_th - th
        e_th0 = e_th
        while e_th > np.pi:
            e_th -= 2 * np.pi
        while e_th <= -np.pi:
            e_th += 2 * np.pi

        u = [
            -k1 * (e[0] * np.cos(th) + e[1] * np.sin(th)),
            k2 * (e_th)
        ]

        e_norm = np.linalg.norm(e)
        if e_norm < abs(u[0]) * self.params['dt']:
            u[0] = np.sign(u[0]) * e_norm / self.params['dt']

        if self.robot.u_min[0] == 0 and u[0] < 0:
            u[0] = 0
            u0_sig = 1
        else:
            u0_sig = u[0] / self.robot.u_max[0] if u[0] >= 0 else u[0] / self.robot.u_min[0]
        u1_sig = u[1] / self.robot.u_max[1] if u[1] >= 0 else u[1] / self.robot.u_min[1]

        sig = max(u0_sig, u1_sig, 1)
        if sig > 1:
            if u0_sig > u1_sig:
                u[0] = self.robot.u_max[0] if u[0] > 0 else self.robot.u_min[0]
                u[1] /= sig
            else:
                u[0] /= sig
                u[1] =  self.robot.u_max[1] if u[1] > 0 else self.robot.u_min[1]

        return u

        # Control law (Assumes Unicycle robot)
        ref_dist = np.linalg.norm(x[:2] - p_ref)
        if ref_dist > 1e-5:
            theta_ref = np.arctan2(p_ref[1] - x[1], p_ref[0] - x[0])
            theta_diff = float(theta_ref - x[-1])
            if theta_diff > np.pi:
                theta_diff -= 2 * np.pi
            if theta_diff < -np.pi:
                theta_diff += 2 * np.pi
            if abs(theta_diff) < 1e-2:  # Less than 0.57 degree error
                # Linear velocity
                u[0] = min(self.robot.u_max[0], ref_dist / self.params['dt'])
            else:
                # Angular velocity
                if theta_diff > 0:
                    u[1] = min(self.robot.u_max[1], theta_diff / self.params['dt'])
                else:
                    u[1] = max(self.robot.u_min[1], theta_diff / self.params['dt'])
        return u

    def r_plus_dist(self):
        return np.linalg.norm(np.array(self.reference_path.interpolate(self.theta).coords[0]) - self.r_plus)

    def theta_plus(self, s):
        return max(self.theta, min(self.theta + s - self.r_plus_dist(), self.theta_g))

    def compute_u(self, x):
        if self.mode == ControlMode.MPC:
            u, w = self.mpc.sol2uds(self.solution)
            return u[:self.robot.nu]
        else:
            # r_kappa = self.ref_kappa(self.robot.h(x), self.rhrp_path[0, :],
            #                          pol2pos(self.path_pol, self.lam_rho, self.mpc.build_params['n_pol']), self.rho)
            r_kappa = self.rhrp_path[0, :]
            if self.robot.__class__.__name__ == 'Unicycle':
                return self.unicycle_sbc(x, r_kappa)
            elif self.robot.__class__.__name__ == 'Omnidirectional':
                k = 100
                u = [k*(r_kappa[0]-x[0]), k*(r_kappa[1]-x[1])]
                for i in range(2):
                    scale = 1
                    if u[i] > self.robot.u_max[i]:
                        scale = self.robot.u_max[i] / u[i]
                    elif u[i] < self.robot.u_min[i]:
                        scale = self.robot.u_min[i] / u[i]
                    u = [scale * u[0], scale * u[1]]
                #
                # u0_sig = u[0] / self.robot.u_max[0]
                # u1_sig = u[1] / self.robot.u_max[1]
                # sig = max(u0_sig, u1_sig, 1)
                # if sig > 1:
                #     if u0_sig > u1_sig:
                #         u[0] = self.robot.u_max[0]
                #         u[1] /= sig
                #     else:
                #         u[0] /= sig
                #         u[1] = self.robot.u_max[1]
                return u
            else:
                print("No SBC for this robot model!!!!")

    def update_policy(self, x, obstacles, workspace=None):
        if workspace is None:
            workspace = Ellipse([1.e10, 1.e10])
        p = self.robot.h(x)

        obstacles_local = obstacles.copy()
        # Adjust for robot radius
        if self.robot.radius > 0:
            obstacles_local, workspace, _, _ = rho_environment(workspace, obstacles_local, self.robot.radius)

        # Adjust for moving obstacles
        if self.params['velocity_obstacle']:
            for i, o in enumerate(obstacles_local):
                if not (o._motion_model is None or o._motion_model.__class__.__name__ == "Static"):
                    o_pol = o.polygon()
                    xvel, yvel = o._motion_model.lin_vel()
                    dil = StarshapedPolygon(shapely.MultiPolygon([o_pol, shapely.affinity.translate(o_pol, xoff=xvel*self.params['dt'], yoff=yvel*self.params['dt'])]).convex_hull)
                    # dil = o.dilated_obstacle(padding=max(abs(o._motion_model.lin_vel())) * self.params['dt'], id="duplicate")
                    if dil.polygon().distance(shapely.geometry.Point(p)) > max(abs(o._motion_model.lin_vel()))*self.params['dt']:
                        obstacles_local[i] = dil
                    else:
                        print("Not dillated obstacle.")
                    # dil = o.dilated_obstacle(padding=max(abs(o._motion_model.lin_vel())) * self.params['dt'], id="duplicate")
                    # _, ax = o.draw()
                    # ax.set_xlim([-1, 12])
                    # ax.set_ylim([-8, 4])
                    # obstacles_local[i].draw(ax=ax, fc='r', alpha=0.4, zorder=-2)
                    # dil.draw(ax=ax, fc='b', alpha=0.4, zorder=-3)
                    # plt.show()

        if self.params['workspace_horizon'] > 0:
            obstacle_detection_region = shapely.geometry.Point(p).buffer(self.params['workspace_horizon'])
            obstacles_filtered = []
            for o in obstacles_local:
                if obstacle_detection_region.intersects(o.polygon()):
                    obstacles_filtered += [o]
            obstacles_local = obstacles_filtered
        else:
            obstacles_filtered = []
            for o in obstacles_local:
                if workspace.polygon().intersects(o.polygon()):
                    obstacles_filtered += [o]
            obstacles_local = obstacles_filtered

        # Initialize rs to robot position
        if self.r_plus is None:
            self.r_plus = p

        self.timing['workspace'] = 0
        self.timing['target'] = 0
        self.timing['mpc'] = 0
        self.timing['workspace_detailed'] = None

        pg_prev = self.pg
        self.rg = None
        self.pg = None
        self.rho = self.params['rho0']
        ref_end = np.array(self.reference_path.coords[-1])
        close_to_convergence = (np.linalg.norm(self.r_plus - ref_end) < self.rho) and (self.theta >= self.theta_g)
        if close_to_convergence:
            self.mode = ControlMode.SBC
            for i in range(len(self.rhrp_s)):
                self.rhrp_path[i, :] = ref_end
        else:
            ds_path_generation = True
            if self.theta_g > 0:
                # Extract receding path from global target path
                t0 = tic()
                obstacles_rho, workspace_rho, free_rho_sh, obstacles_rho_sh = rho_environment(workspace, obstacles_local, self.rho)
                self.timing['workspace'] = toc(t0)

                t0 = tic()
                nominal_rhrp_sh, nominal_rhrp_collision_free = self.nominal_rhrp(self.r_plus, free_rho_sh)
                if nominal_rhrp_collision_free:
                    ds_path_generation = False
                    if nominal_rhrp_sh.length > 0:
                        self.rhrp_path = np.array([nominal_rhrp_sh.interpolate(s).coords[0] for s in self.rhrp_s])
                    else:
                        self.rhrp_path = np.tile(nominal_rhrp_sh.coords[0], (len(self.rhrp_s), 1))
                    self.obstacles_rho = obstacles_rho
                    self.workspace_rho = workspace_rho
                    self.obstacles_star = []
                    self.obstacle_clusters = None
                    self.pg = self.rhrp_path[-1, :]
                    rhrp_path_length = self.rhrp_L
                self.timing['target'] = toc(t0)

            if ds_path_generation:
                # Find attractor for DS dynamics
                if self.theta_g == 0:
                    self.pg = np.array(self.reference_path.coords[0])
                else:
                    t0 = tic()
                    self.theta = self.theta_plus(self.params['nominal_rhrp_horizon'])
                    for theta in np.arange(self.theta_plus(self.params['nominal_rhrp_horizon']), self.theta_g, np.diff(self.rhrp_s[:2])):
                        # if self.reference_path.interpolate(theta).disjoint(obstacles_rho_sh):
                        if self.reference_path.interpolate(theta).within(free_rho_sh):
                            self.theta = theta
                            break
                    self.pg = np.array(self.reference_path.interpolate(self.theta).coords[0])
                    print(self.theta_g, theta, self.reference_path.interpolate(theta).within(free_rho_sh))
                    self.timing['workspace'] += toc(t0)

                pg_buffer_thresh = 0.5
                # new_pg_buffer = not self.ds_generated_path or np.linalg.norm(self.pg_buffer - self.pg) > pg_buffer_thresh
                # if new_pg_buffer:
                #     self.pg_buffer = self.pg

                buffer_active = self.params['buffer'] and (self.ds_generated_path and np.linalg.norm(pg_prev - self.pg) < pg_buffer_thresh)
                buffer_path = self.rhrp_path if buffer_active else None
                if buffer_active:
                    self.pg_buffer = self.pg

                local_pars = self.params.copy()
                local_pars['max_rhrp_compute_time'] -= self.timing['target']

                self.rhrp_path, rhrp_path_length, self.obstacle_clusters, self.obstacles_rho, self.workspace_rho, \
                self.obstacles_star, self.rho, self.rg, workspace_exitflag, workspace_timing, target_timing, self.timing['workspace_detailed'] = \
                    ds_path_gen(obstacles_local, workspace, p, self.pg, self.r_plus, self.params['rho0'], self.params['hull_epsilon'], self.rhrp_s,
                                self.obstacle_clusters, buffer_path, local_pars, self.verbosity)

                # Update O+
                t0 = tic()
                obstacles_rho_sh = shapely.ops.unary_union([o.polygon() for o in self.obstacles_rho])
                C_rho0 = self.workspace_rho.polygon().difference(obstacles_rho_sh).buffer(self.params['rho0'])
                env_DSW = workspace_exitflag != 0
                if not (env_DSW and (shapely.geometry.Point(p).within(C_rho0) or self.rho == self.params['rho0'])):
                    self.obstacle_clusters = None
                self.timing['workspace'] += toc(t0)

                self.timing['workspace'] += workspace_timing
                self.timing['target'] += target_timing

            self.ds_generated_path = ds_path_generation

            t0 = tic()
            # Approximate target with polynomials
            self.path_pol = np.polyfit(self.rhrp_s, self.rhrp_path[:, 0], self.params['n_pol']).tolist() + \
                            np.polyfit(self.rhrp_s, self.rhrp_path[:, 1], self.params['n_pol']).tolist()
            # Force init position to be correct
            self.path_pol[self.params['n_pol']] = self.rhrp_path[0, 0]
            self.path_pol[-1] = self.rhrp_path[0, 1]
            # Compute polyfit approximation error
            self.epsilon = max(np.linalg.norm(self.rhrp_path - np.array([pol2pos(self.path_pol, s, self.mpc.build_params['n_pol']) for s in self.rhrp_s]), axis=1))
            self.timing['target'] += toc(t0)

            t0 = tic()
            self.lam_rho = self.params['lambda'] * self.rho  # Parameter for tracking error constraint

            # Compute MPC solution
            e_max = self.rho - self.epsilon
            solution_data = self.mpc.run(x.tolist(), self.u_prev, self.path_pol, self.params, e_max, self.lam_rho, self.solution, verbosity=self.verbosity)
            if solution_data is None:
                self.solution, self.mpc_exit_status, self.sol_feasible = None, "BaseSolution", False
            else:
                self.solution, self.mpc_exit_status = solution_data.solution, solution_data.exit_status
                self.sol_feasible = self.mpc.is_feasible(self.solution, x.tolist(), self.path_pol, e_max, self.lam_rho, d=self.verbosity > 0)
            # self.solution, self.sol_feasible, self.mpc_exit_status = self.mpc.run(x.tolist(), self.u_prev, self.path_pol, self.params,
            #                                                e_max, self.lam_rho, verbosity=self.verbosity)
            self.timing['mpc'] = toc(t0)

            if (np.linalg.norm(self.rhrp_path[-1, :]-self.pg) <= 0.1 * self.rhrp_L or rhrp_path_length > 0.1 * self.rhrp_L) and self.sol_feasible:
                self.mode = ControlMode.MPC
            else:
                if self.verbosity > 0:
                    print("[Motion Controller]: MPC solution not feasible. Using default control law.")
                self.mode = ControlMode.SBC

        if self.mode == ControlMode.SBC:
            # r_kappa = self.ref_kappa(p, self.rhrp_path[0, :], pol2pos(self.path_pol, self.lam_rho, self.mpc.build_params['n_pol']), self.rho)
            # r_kappa = self.rhrp_path[0, :]
            # u = self.unicycle_sbc(x, r_kappa)
            # if self.verbosity > 0:
            #     print("[Motion Controller]: MPC solution not feasible. Using default control law. u = " + str(u))
            self.r_plus = self.rhrp_path[0, :]
        else:
            u, w = self.mpc.sol2uds(self.solution)
            # u = u[:self.robot.nu]
            # Update rs and theta

            # if np.linalg.norm(self.rhrp_path[0, :] - np.array(self.reference_path.interpolate(self.theta).coords[0])) < self.rho:
            if self.theta_g > 0 and self.r_plus_dist() < self.rho:
                self.theta = min(self.theta + w[0] * self.params['dt'], self.theta_g)

            self.r_plus = np.array(pol2pos(self.path_pol, w[0] * self.params['dt'], self.mpc.build_params['n_pol']))
            # if update_theta:
            # if self.theta < self.theta_g and np.linalg.norm(self.rhrp_path[0, :] - np.array(self.reference_path.interpolate(self.theta).coords[0])) < self.rho:
            #     self.theta += w[0] * self.params['dt']
        # print(np.linalg.norm(self.r_plus-self.rhrp_path[0, :]) )


        # return self.sppc(x, pol2pos(self.path_pol, self.lam_rho, self.mpc.build_params['n_pol']))
        # return self.sppc(x, self.reference_path.coords[-1])

        # self.sol_feasible = False

        self.u_prev = self.compute_u(x)
        # return np.array(u)


def ds_path_gen(obstacles, workspace, p, pg, r_plus, rho0, hull_epsilon, rhrp_s,
                previous_obstacle_clusters=None, buffer_path=None, params=None, verbosity=0):
    # Update obstacles
    obstacle_clusters, r0, rg, rho, obstacles_rho, workspace_rho, workspace_timing, workspace_timing_detailed, workspace_exitflag = \
        workspace_modification(obstacles, workspace, p, pg, r_plus, rho0, hull_epsilon, previous_obstacle_clusters, params, verbosity)
    obstacles_star = [o.cluster_obstacle for o in obstacle_clusters]
    # if exit_flag == 0:
    #     obstacle_clusters = None

    # Make sure all polygon representations are computed
    [o._compute_polygon_representation() for o in obstacles_star]

    # Buffer target path
    init_path, init_s = None, None
    if buffer_path is not None:
        # Shift path to start closest to current r_plus
        init_path = buffer_path[np.argmin(np.linalg.norm(r0 - buffer_path, axis=1)):, :]
        if np.linalg.norm(r0 - init_path[0, :]) > 1e-6:
            init_path = np.vstack((r0, init_path))

        # ||self.r_plus - p|| < rho from construction
        for r in init_path:
            # NOTE: Assumes previous path not outside workspace due to static workspace
            if not all([o.exterior_point(r) for o in obstacles_star]) or not workspace_rho.interior_point(r):
            # if not all([o.exterior_point(r) for o in obstacles_star]):
                if verbosity > 1:
                    print("[Path Generator]: No reuse of previous path. Path not collision-free.")
                init_path = None
                break

        if init_path is not None:
            # Cut off stand still padding in init path
            init_path_stepsize = np.linalg.norm(np.diff(init_path, axis=0), axis=1)
            init_s = np.hstack((0, np.cumsum(init_path_stepsize)))
            init_path_mask = [True] + (init_path_stepsize > 1e-8).tolist()
            init_path = init_path[init_path_mask, :]
            init_s = init_s[init_path_mask]


    # Update target path
    rhrp_path, rhrp_path_length, path_timing = \
        path_generator(r0, rg, obstacles_star, workspace_rho, rhrp_s, init_path, init_s, params, verbosity)

    return rhrp_path, rhrp_path_length, obstacle_clusters, obstacles_rho, workspace_rho, obstacles_star, rho, rg, workspace_exitflag, workspace_timing, path_timing, workspace_timing_detailed