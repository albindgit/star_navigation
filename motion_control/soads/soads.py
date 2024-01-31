import numpy as np
import shapely
from starworlds.starshaped_hull import cluster_and_starify
from starworlds.utils.misc import tic, toc

class SoadsController:

    def __init__(self, params, robot, verbose=False):
        if not robot.__class__.__name__ == 'Omnidirectional':
            raise NotImplementedError("SoadsController only implemented for Omnidirectional robots.")
        self.params = params
        self.robot = robot
        self.obstacle_clusters = None
        self.obstacles_star = []
        self.dp_prev = None
        self.verbose = verbose
        self.timing = {'obstacle': 0, 'control': 0}

    def compute_u(self, x, pg, obstacles, workspace=None):
        p = self.robot.h(x)

        if self.params['starify']:
            self.obstacle_clusters, obstacle_timing, exit_flag, n_iter = cluster_and_starify(obstacles, p, pg,
                                                                                             self.params['hull_epsilon'],
                                                                                             max_compute_time=  self.params['max_compute_time'],
                                                                                             workspace=workspace,
                                                                                             previous_clusters=self.obstacle_clusters,
                                                                                             # dx_prev=self.dp_prev,
                                                                                             make_convex=self.params['make_convex'],
                                                                                             verbose=self.verbose)
            self.timing['obstacle'] = sum(obstacle_timing)
            self.obstacles_star = [cl.cluster_obstacle for cl in self.obstacle_clusters]

            # import matplotlib.pyplot as plt
            # tmp_h = []
            # ax = plt.gca()
            # for o in self.obstacle_clusters:
            #     for k in o.kernel_points:
            #         tmp_h += ax.plot(*k, 'y*')
            # while not plt.waitforbuttonpress(): pass
            # [h.remove() for h in tmp_h]


            # print(obstacle_timing)

            # self.obstacles_star = []
            # for cl in self.obstacle_clusters:
            #     cl_obstacles = cl.obstacles
            #     for i in range(len(cl_obstacles)):
            #         cl_obstacles[i].set_xr(cl.cluster_obstacle.xr())
            #     self.obstacles_star += cl_obstacles
            # print(obstacle_timing, n_iter)
            # print("(Cl/Adm/Hull) calculation ({:.2f}/{:.2f}/{:.2f}) [{:.0f}]".format(*obstacle_timing[:-1], n_iter))

        else:
            self.timing['obstacle'] += 0
            self.obstacles_star = obstacles

        # if any([o.interior_point(p) for o in self.obstacles_star]):
        #     import matplotlib.pyplot as plt
        #     from obstacles.tools import draw_shapely_polygon, is_cw, is_ccw, is_collinear, RayCone
        #     from starshaped_hull import admissible_kernel
        #     print("p in obstacles")
        #     for cl in self.obstacle_clusters:
        #         _, ax = plt.subplots()
        #         # cl.cluster_obstacle.draw(ax=ax, fc='r', alpha=0.5)
        #         for o in cl.obstacles:
        #             o.draw(ax=ax, show_name=1)
        #             # if o.id() == 1:
        #             #     ad_ker = admissible_kernel(o, p)
        #             #     # tp1, tp2 = o.tangent_points(p)
        #             #     # if is_ccw(p, tp1, tp2):
        #             #     #     print('ccw')
        #             #     # elif is_cw(p, tp1, tp2):
        #             #     #     print('cw')
        #             #     # elif is_collinear(p, tp1, tp2):
        #             #     #     print('collinear')
        #             #     # ax.plot(*tp1, 'rx')
        #             #     # ax.plot(*tp2, 'rs')
        #             #     draw_shapely_polygon(ad_ker.polygon(), ax=ax, fc='y', alpha=0.3)
        #
        #         ax.plot(*p, 'o')
        #         ax.set_xlim([0, 13])
        #         ax.set_ylim([-1, 11])
        #         adm_ker_robot = RayCone.intersection_same_apex([admissible_kernel(o, p) for o in cl.obstacles], debug_info={'o': obstacles, 'x': p, 'xlim': [0, 13], 'ylim': [-1, 11]})
        #         if cl.admissible_kernel is not None:
        #             # draw_shapely_polygon(cl.admissible_kernel, ax=ax, fc='y', alpha=0.3)
        #             draw_shapely_polygon(adm_ker_robot.polygon(), ax=ax, fc='y', alpha=0.3)
        #     plt.show()

        t0 = tic()
        dist_to_goal = np.linalg.norm(p - pg)
        if self.params['lin_vel'] > 0:
            lin_vel = min(self.params['lin_vel'], dist_to_goal / self.params['dt'])
            unit_mag = True
        else:
            lin_vel = 1
            unit_mag = False
        dp = lin_vel * f(p, pg, self.obstacles_star, workspace=workspace, unit_magnitude=unit_mag, crep=self.params['crep'],
                         reactivity=self.params['reactivity'], tail_effect=self.params['tail_effect'],
                         adapt_obstacle_velocity=self.params['adapt_obstacle_velocity'],
                         convergence_tolerance=self.params['convergence_tolerance'])

        p_next = p + dp * self.params['dt']

        while not all([o.exterior_point(p_next) for o in self.obstacles_star]) and (workspace is None or workspace.interior_point(p_next)):
            dp *= self.params['dp_decay_rate']
            p_next = p + dp * self.params['dt']
            # Additional compute time check
            if toc(t0) > self.params['max_compute_time']:
                if self.verbose or True:
                    print("[Max compute time in soads when adjusting for collision. ")
                dp *= 0
                break
        self.timing['control'] = toc(t0)

        self.dp_prev = dp

        return dp

def f_nominal(r, f_nominal, obstacles, workspace=None, adapt_obstacle_velocity=False, unit_magnitude=False, crep=1., reactivity=1., tail_effect=False,
      convergence_tolerance=1e-4, d=False):
    No = len(obstacles)

    if No == 0 and workspace is None:
        return f_nominal

    ef = [-f_nominal[1], f_nominal[0]]
    Rf = np.vstack((f_nominal, ef)).T

    r_sh = shapely.geometry.Point(r)

    mu = [obs.reference_direction(r) for obs in obstacles]
    normal = [obs.normal(r) for obs in obstacles]
    gamma = [obs.distance_function(r) for obs in obstacles]
    dist = [obs.polygon().distance(r_sh) for obs in obstacles]

    workspace_avoidance = False
    if workspace is not None:
        workspace_avoidance = np.linalg.norm(
            workspace.xr() - r) > 1e-4  # workspace.polygon().exterior.distance(r_sh) < 1

    if workspace_avoidance:
        mu += [workspace.reference_direction(r)]
        normal += [workspace.normal(r)]
        gamma += [1 / workspace.distance_function(r)]
        dist += [workspace.polygon().exterior.distance(r_sh)]
        No += 1

    # Compute weights
    # w = compute_weights(gamma)
    w = compute_weights(dist, gamma_lowerlimit=0)

    # Compute obstacle velocities
    xd_o = np.zeros((2, No))
    if adapt_obstacle_velocity:
        for i, obs in enumerate(obstacles):
            xd_o[:, i] = obs.vel_intertial_frame(r)

    kappa = 0.
    f_mag = 0.
    for i in range(No):
        # Compute basis matrix
        E = np.zeros((2, 2))
        E[:, 0] = mu[i]
        E[:, 1] = [-normal[i][1], normal[i][0]]
        # Compute eigenvalues
        D = np.zeros((2, 2))
        if workspace_avoidance and i == No - 1:
            inactive_repulsion = normal[i].dot(f_nominal) < 0
            # inactive_repulsion = False
            D[0, 0] = 1 if inactive_repulsion else 1 - crep / (gamma[i] ** (1 / reactivity))
            D[1, 1] = 1 + 1 / gamma[i] ** (1 / reactivity)
        else:
            inactive_repulsion = mu[i].dot(f_nominal) > 0 and normal[i].dot(f_nominal) > 0
            # inactive_repulsion = False
            D[0, 0] = 1 if inactive_repulsion else 1 - crep / (gamma[i] ** (1 / reactivity))
            D[1, 1] = 1 + 1 / gamma[i] ** (1 / reactivity)
        # Compute modulation
        M = E.dot(D).dot(np.linalg.inv(E))
        # f_i = M.dot(f_nominal)
        f_i = M.dot(f_nominal - xd_o[:, i]) + xd_o[:, i]
        # Compute contribution to velocity magnitude
        f_i_abs = np.linalg.norm(f_i)
        f_mag += w[i] * f_i_abs
        # Compute contribution to velocity direction
        nu_i = f_i / f_i_abs
        nu_i_hat = Rf.T.dot(nu_i)
        kappa_i = np.arccos(np.clip(nu_i_hat[0], -1, 1)) * np.sign(nu_i_hat[1])
        kappa += w[i] * kappa_i
    kappa_norm = abs(kappa)
    f_o = Rf.dot([np.cos(kappa_norm), np.sin(kappa_norm) / kappa_norm * kappa]) if kappa_norm > 0. else f_nominal

    if unit_magnitude:
        f_mag = 1.

    # f_mag = np.linalg.norm(f_nominal)

    return f_mag * f_o

# TODO: Rename to something with linear attractor
def f(r, rg, obstacles, workspace=None, adapt_obstacle_velocity=False, unit_magnitude=False, crep=1., reactivity=1., tail_effect=False,
      convergence_tolerance=1e-4, d=False):
    goal_vector = rg - r
    goal_dist = np.linalg.norm(goal_vector)
    if goal_dist < convergence_tolerance:
        return 0 * r

    No = len(obstacles)
    fa = goal_vector / goal_dist  # Attractor dynamics


    if No == 0 and workspace is None:
        return fa

    return f_nominal(r, fa, obstacles, workspace, adapt_obstacle_velocity, unit_magnitude, crep, reactivity, tail_effect,
      convergence_tolerance, d)

    ef = [-fa[1], fa[0]]
    Rf = np.vstack((fa, ef)).T

    for o in obstacles:
        if o.boundary_mapping(r) is None:
            import matplotlib.pyplot as plt
            o.draw(ax=plt.gca(), fc='r', alpha=0.8)
            hs = plt.plot(*zip(o.xr(),r), 'b-')
            hs += plt.plot(*zip(o.xr(), o.xr()+1.1*o.enclosing_ball_diameter*o.reference_direction(r)), 'r--')
            while not plt.waitforbuttonpress(): pass
            [h.remove() for h in hs]

    r_sh = shapely.geometry.Point(r)

    mu = [obs.reference_direction(r) for obs in obstacles]
    normal = [obs.normal(r) for obs in obstacles]
    gamma = [obs.distance_function(r) for obs in obstacles]
    dist = [obs.polygon().distance(r_sh) for obs in obstacles]

    workspace_avoidance = False
    if workspace is not None:
        workspace_avoidance = np.linalg.norm(workspace.xr()-r) > 1e-4# workspace.polygon().exterior.distance(r_sh) < 1

    if workspace_avoidance:
        mu += [workspace.reference_direction(r)]
        normal += [workspace.normal(r)]
        gamma += [1 / workspace.distance_function(r)]
        dist += [workspace.polygon().exterior.distance(r_sh)]
        No += 1

    # Compute weights
    # w = compute_weights(gamma)
    w = compute_weights(dist, gamma_lowerlimit=0)

    # Compute obstacle velocities
    xd_o = np.zeros((2, No))
    if adapt_obstacle_velocity:
        for i, obs in enumerate(obstacles):
            xd_o[:, i] = obs.vel_intertial_frame(r)

    kappa = 0.
    f_mag = 0.
    for i in range(No):
        # Compute basis matrix
        E = np.zeros((2, 2))
        E[:, 0] = mu[i]
        E[:, 1] = [-normal[i][1], normal[i][0]]
        # Compute eigenvalues
        D = np.zeros((2, 2))
        if workspace_avoidance and i == No-1:
            inactive_repulsion = normal[i].dot(fa) < 0
            D[0, 0] = 1 if inactive_repulsion else 1 - crep / (gamma[i] ** (1 / reactivity))
            D[1, 1] = 1 + 1 / gamma[i] ** (1 / reactivity)
        else:
            inactive_repulsion = mu[i].dot(fa) > 0 and normal[i].dot(fa) > 0
            D[0, 0] = 1 if inactive_repulsion else 1 - crep / (gamma[i] ** (1 / reactivity))
            D[1, 1] = 1 + 1 / gamma[i] ** (1 / reactivity)
        # Compute modulation
        M = E.dot(D).dot(np.linalg.inv(E))
        # f_i = M.dot(fa)
        f_i = M.dot(fa - xd_o[:, i]) + xd_o[:, i]
        # Compute contribution to velocity magnitude
        f_i_abs = np.linalg.norm(f_i)
        f_mag += w[i] * f_i_abs
        # Compute contribution to velocity direction
        nu_i = f_i / f_i_abs
        nu_i_hat = Rf.T.dot(nu_i)
        kappa_i = np.arccos(np.clip(nu_i_hat[0], -1, 1)) * np.sign(nu_i_hat[1])
        kappa += w[i] * kappa_i
    kappa_norm = abs(kappa)
    f_o = Rf.dot([np.cos(kappa_norm), np.sin(kappa_norm) / kappa_norm * kappa]) if kappa_norm > 0. else fa

    if unit_magnitude:
        f_mag = 1.
    return f_mag * f_o


def compute_weights(
    gamma,
    gamma_lowerlimit=1,
    gamma_upperlimit=500,
    weightPow=2
):
    """Compute weights based on a distance measure (with no upper limit)"""
    gamma = np.array(gamma)
    n_points = gamma.shape[0]

    collision_points = gamma <= gamma_lowerlimit

    if np.any(collision_points):  # at least one
        if np.sum(collision_points) == 1:
            return collision_points * 1.0
        else:
            # TODO: continuous weighting function
            w = collision_points * 1.0 / np.sum(collision_points)
            return w

    gamma = gamma - gamma_lowerlimit

    w = (1 / gamma) ** weightPow
    if np.sum(w) == 0:
        return w
    w = w / np.sum(w)  # Normalization
    return w


def rollout(x, xg, obstacles, step_size=0.01, max_nr_steps=10000, max_distance=np.inf, convergence_threshold=0.05, crep=1., reactivity=1.):
    xs = np.zeros((2, max_nr_steps))
    xs[:, 0] = x
    dist_travelled = 0
    for k in range(1, max_nr_steps):
        dist_to_goal = np.linalg.norm(xs[:, k-1]-xg)
        if dist_to_goal < convergence_threshold or dist_travelled > max_distance:
            return xs[:, :k]
        else:
            step_mag = min(step_size, dist_to_goal)
            step_dir = f(xs[:, k-1], xg, obstacles, unit_magnitude=1, crep=crep, reactivity=reactivity)
            xs[:, k] = xs[:, k-1] + step_mag * step_dir
            while any([o.interior_point(xs[:, k]) for o in obstacles]):
                step_mag /= 2.
                xs[:, k] = xs[:, k-1] + step_mag * step_dir
            dist_travelled += step_mag
    return xs


def draw_streamlines(xg, obstacles, ax, init_pos_list, step_size=0.01, max_nr_steps=10000, max_distance=np.inf,
                     convergence_threshold=0.05, crep=1., reactivity=1.,
                     arrow_step=0.5, color='k', linewidth=1, **kwargs):
    for x0 in init_pos_list:
        xs = rollout(x0, xg, obstacles, step_size, max_nr_steps, max_distance, convergence_threshold, crep, reactivity)
        travelled_dist = np.cumsum(np.linalg.norm(np.diff(xs, axis=1), axis=0))
        travelled_dist = np.insert(travelled_dist, 0, 0)
        arrow_dist = np.arange(0, travelled_dist[-1], arrow_step)
        arrows_X = np.interp(arrow_dist, travelled_dist, xs[0, :])
        arrows_Y = np.interp(arrow_dist, travelled_dist, xs[1, :])
        arrows_U = np.zeros_like(arrows_X)
        arrows_V = np.zeros_like(arrows_X)
        # ax.plot(arrows_X, arrows_Y, 'k*')
        for j in range(len(arrow_dist)):
            arrows_U[j], arrows_V[j] = f(np.array([arrows_X[j], arrows_Y[j]]), xg, obstacles, unit_magnitude=1,
                                         crep=crep, reactivity=reactivity)
        ax.quiver(arrows_X, arrows_Y, arrows_U, arrows_V, color=color, scale=50, width=0.005, scale_units='width', headlength=2, headaxislength=2)
        # ax.plot(xs[0, :], xs[1, :], color=color, linewidth=linewidth, **kwargs)

def draw_vector_field(xg, obstacles, ax, workspace=None, xlim=None, ylim=None, n=50, crep=1., reactivity=1., **kwargs):
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    def in_boundary(x):
        return True if workspace is None else workspace.interior_point(x)

    Y, X = np.mgrid[ylim[0]:ylim[1]:n*1j, xlim[0]:xlim[1]:n*1j]
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)
    mask = np.zeros(U.shape, dtype=bool)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            x = np.array([X[i, j], Y[i, j]])
            if in_boundary(x) and np.all([o.exterior_point(x) for o in obstacles]):
                U[i, j], V[i, j] = f(x, xg, obstacles, workspace=workspace, unit_magnitude=1, crep=crep, reactivity=reactivity)
            else:
                mask[i, j] = True
                U[i, j] = np.nan
    U = np.ma.array(U, mask=mask)
    # return ax.quiver(X, Y, U, V)
    return ax.streamplot(X, Y, U, V, **kwargs)