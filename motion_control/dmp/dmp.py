import numpy as np
import shapely
from starworlds.starshaped_hull import cluster_and_starify
from starworlds.utils.misc import tic, toc
from motion_control.soads import f_nominal as soads_f

class DMP:

    def __init__(self, parameters, reference_path, path_speed):
        self.K = parameters['K']
        self.D = parameters['D']
        self.alpha_s = None
        self.n_bfs = parameters['n_bfs']
        self.n = 2 # Hardcoded 2D path
        self.goal = np.array(reference_path.coords[-1])
        self.p0 = np.array(reference_path.coords[0])
        self.centers = None
        self.widths = None
        self.w = None
        self.fit(reference_path, path_speed)
        self.z0 = np.concatenate((self.p0, 0 * self.p0))

    def p(self, z):
        assert len(z) == 2*self.n
        return z[:self.n]

    def v(self, z, obstacles=None, workspace=None, adapt_obstacle_velocity=False, unit_magnitude=False, crep=1., reactivity=1., tail_effect=False,
      convergence_tolerance=1e-4):
        assert len(z) == 2*self.n
        if obstacles is None and workspace is None:
            return z[self.n:]
        else:
            return soads_f(z[:self.n], z[self.n:], obstacles, workspace, adapt_obstacle_velocity, unit_magnitude, crep, reactivity, tail_effect,
      convergence_tolerance)

    def f(self, z, t, obstacles=None, workspace=None, adapt_obstacle_velocity=False, unit_magnitude=False, crep=1., reactivity=1., tail_effect=False,
      convergence_tolerance=1e-4):
        s = np.exp(-self.alpha_s * t)
        psi = self.basis_fcn(s)
        fe = self.w.dot(psi) / np.maximum(np.sum(psi), 1e-8) * (self.goal - self.p0) * s
        return self.K * (self.goal - self.p(z)) - self.D * self.v(z, obstacles, workspace, adapt_obstacle_velocity, unit_magnitude, crep, reactivity, tail_effect,
      convergence_tolerance) + fe

    def dz(self, z, t, obstacles=None, workspace=None, adapt_obstacle_velocity=False, unit_magnitude=False, crep=1., reactivity=1., tail_effect=False,
      convergence_tolerance=1e-4):
      #   return np.concatenate((self.v(z, obstacles, workspace, adapt_obstacle_velocity, unit_magnitude, crep, reactivity, tail_effect,
      # convergence_tolerance), self.f(z, t)))
        return np.concatenate((self.v(z, obstacles, workspace, adapt_obstacle_velocity, unit_magnitude, crep, reactivity, tail_effect,
      convergence_tolerance), self.f(z, t, obstacles, workspace, adapt_obstacle_velocity, unit_magnitude, crep, reactivity, tail_effect,
      convergence_tolerance)))

    def basis_fcn(self, s, i=None):
        if i is not None:
            return np.exp(-1 / (2 * self.widths[i] ** 2) * (s - self.centers[i]) ** 2)
        if i is None:
            return np.exp(-1 / (2 * self.widths ** 2) * (s - self.centers) ** 2)

    def fit(self, reference_path, path_speed):
        delta_p = int(reference_path.length / path_speed)
        t_demo = np.linspace(0, reference_path.length, 10000)
        self.alpha_s = 1. / t_demo[-1]

        p_demo = np.array([reference_path.interpolate(s).coords[0] for s in t_demo]).T
        
        # Set basis functions
        t_centers = np.linspace(0, reference_path.length, self.n_bfs, endpoint=True)
        self.centers = np.exp(-self.alpha_s * t_centers)
        widths = np.abs((np.diff(self.centers)))
        self.widths = np.concatenate((widths, [widths[-1]]))

        # Calculate derivatives
        dp_demo = (p_demo[:, 1:] - p_demo[:, :-1]) / (t_demo[1:] - t_demo[:-1])
        dp_demo = np.concatenate((dp_demo, np.zeros((self.n, 1))), axis=1)
        ddp_demo = (dp_demo[:, 1:] - dp_demo[:, :-1]) / (t_demo[1:] - t_demo[:-1])
        ddp_demo = np.concatenate((ddp_demo, np.zeros((self.n, 1))), axis=1)

        # Compute weights
        x_seq = np.exp(-self.alpha_s * t_demo)
        self.w = np.zeros((self.n, self.n_bfs))
        for i in range(self.n):
            if abs(self.goal[i] - self.p0[i]) < 1e-5:
                continue
            f_gain = x_seq * (self.goal[i] - self.p0[i])
            f_target = ddp_demo[i, :] - self.K * (self.goal[i] - p_demo[i, :]) + self.D * dp_demo[i, :]
            for j in range(self.n_bfs):
                psi_j = self.basis_fcn(x_seq, j)
                num = f_gain.dot((psi_j * f_target).T)
                den = f_gain.dot((psi_j * f_gain).T)
                if abs(den) < 1e-14:
                    continue
                self.w[i, j] = num / den


class DMPController:

    def __init__(self, params, robot, reference_path, verbose=False):
        if not robot.__class__.__name__ == 'Omnidirectional':
            raise NotImplementedError("DMPController only implemented for Omnidirectional robots.")
        self.params = params
        self.robot = robot
        self.obstacle_clusters = None
        self.obstacles_star = []
        self.dp_prev = None
        self.verbose = verbose
        self.reference_path = shapely.geometry.LineString(reference_path)
        self.theta_g = self.reference_path.length
        self.dmp = DMP(params, self.reference_path, robot.vmax)
        self.z = self.dmp.z0.copy()
        self.u = None
        self.theta = 0
        self.timing = {'workspace': 0, 'target': 0, 'mpc': 0}

    def compute_u(self, x):
        return self.v_ref #+ 1 * (self.p_ref - self.robot.h(x))

    def update_policy(self, x, obstacles, workspace=None):
        p = self.robot.h(x)

        # if self.params['starify']:
        #     self.obstacle_clusters, obstacle_timing, exit_flag, n_iter = cluster_and_starify(obstacles, p, pg,
        #                                                                                      self.params['hull_epsilon'],
        #                                                                                      max_compute_time=  self.params['max_compute_time'],
        #                                                                                      workspace=workspace,
        #                                                                                      previous_clusters=self.obstacle_clusters,
        #                                                                                      # dx_prev=self.dp_prev,
        #                                                                                      make_convex=self.params['make_convex'],
        #                                                                                      verbose=self.verbose)
        #     self.timing['obstacle'] = sum(obstacle_timing)
        #     self.obstacles_star = [cl.cluster_obstacle for cl in self.obstacle_clusters]
        # else:
        #     self.timing['obstacle'] += 0
        #     self.obstacles_star = obstacles

        # t0 = tic()
        # dist_to_goal = np.linalg.norm(p - pg)

        dtheta_dt = 0.2

        # obstacles=None
        self.v_ref = self.dmp.v(self.z, obstacles, workspace, unit_magnitude=False, crep=self.params['crep'],
                     reactivity=self.params['reactivity'], tail_effect=self.params['tail_effect'],
                     adapt_obstacle_velocity=self.params['adapt_obstacle_velocity'],
                     convergence_tolerance=self.params['convergence_tolerance']) * dtheta_dt
        self.p_ref = self.dmp.p(self.z)

        dv = (self.v_ref - self.z[2:]) / self.params['dt']

        dz = np.concatenate((self.v_ref, dv))
        dz = self.dmp.dz(self.z, self.theta, obstacles, workspace, unit_magnitude=False, crep=self.params['crep'],
                     reactivity=self.params['reactivity'], tail_effect=self.params['tail_effect'],
                     adapt_obstacle_velocity=self.params['adapt_obstacle_velocity'],
                     convergence_tolerance=self.params['convergence_tolerance'])

        # Target state integration
        # dz = np.concatenate((self.dmp.v(self.z), self.dmp.f(self.z, self.theta)))
        self.z += dz * dtheta_dt * self.params['dt']

        self.theta += dtheta_dt * self.params['dt']



        # p_next = p + dp * self.params['dt']
        #
        # while not all([o.exterior_point(p_next) for o in self.obstacles_star]) and (workspace is None or workspace.interior_point(p_next)):
        #     dp *= self.params['dp_decay_rate']
        #     p_next = p + dp * self.params['dt']
        #     # Additional compute time check
        #     if toc(t0) > self.params['max_compute_time']:
        #         if self.verbose or True:
        #             print("[Max compute time in soads when adjusting for collision. ")
        #         dp *= 0
        #         break
        # self.timing['control'] = toc(t0)
        #
        # self.dp_prev = dp
