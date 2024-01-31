import numpy as np
from starworlds.utils.misc import tic, toc
from motion_control.soads.soads import f as soads_f


def path_generator(r0, rg, obstacles, workspace, path_s, init_path=None, init_s=None, params=None, verbosity=0):
    par = {'ds_decay_rate': 0.5, 'ds_increase_rate': 2., 'max_nr_steps': 1000, 'convergence_tolerance': 1e-5,
           'reactivity': 1., 'crep': 1., 'tail_effect': False, 'max_rhrp_compute_time': np.inf}
    for k in par.keys():
        if k in params:
            par[k] = params[k]

    # Assumes equidistant path_s
    path_ds = path_s[1] - path_s[0]

    t0 = tic()

    # Initialize
    ds = path_ds
    s = np.zeros(par['max_nr_steps'])
    r = np.zeros((par['max_nr_steps'], r0.size))
    if init_path is not None:
        i = init_path.shape[0]
        r[:i, :] = init_path
        s[:i] = init_s
    else:
        i = 1
        r[0, :] = r0


    L = path_s[-1]

    while True:
        dist_to_goal = np.linalg.norm(r[i - 1, :] - rg)
        # Check exit conditions
        if dist_to_goal < par['convergence_tolerance']:
            if verbosity > 2:
                print("[Path Generator]: Path converged. " + str(
                    int(100 * (s[i - 1] / L))) + "% of path completed.")
            break
        if s[i - 1] >= 0.5 * L:
            if verbosity > 2:
                print("[Path Generator]: Completed path length. " + str(
                    int(100 * (s[i - 1] / L))) + "% of path completed.")
            break
        if toc(t0) > par['max_rhrp_compute_time']:
            if verbosity > 0:
                print("[Path Generator]: Max compute time in path integrator. " + str(
                    int(100 * (s[i - 1] / L))) + "% of path completed.")
            break
        if i >= par['max_nr_steps']:
            if verbosity > 0:
                print("[Path Generator]: Max steps taken in path integrator. " + str(
                    int(100 * (s[i - 1] / L))) + "% of path completed.")
            break

        # Movement using SOADS dynamics
        dr = min(1, dist_to_goal) * soads_f(r[i - 1, :], rg, obstacles, workspace=workspace, adapt_obstacle_velocity=False,
                                            unit_magnitude=True, crep=par['crep'],
                                            reactivity=par['reactivity'], tail_effect=par['tail_effect'],
                                            convergence_tolerance=par['convergence_tolerance'])
        r[i, :] = r[i - 1, :] + dr * ds
        ri_collision = False

        def collision_free(p):
            return all([o.exterior_point(p) for o in obstacles]) and workspace.interior_point(p)

        # Adjust for collisions that may arise due to crep!=1
        if not collision_free(r[i, :]):
            dr = min(1, dist_to_goal) * soads_f(r[i - 1, :], rg, obstacles, workspace=workspace,
                                                adapt_obstacle_velocity=False,
                                                unit_magnitude=True, crep=1,
                                                reactivity=1, convergence_tolerance=par['convergence_tolerance'])
            r[i, :] = r[i - 1, :] + dr * ds

            # while any([o.interior_point(r[i, :]) for o in obstacles]):
            while not collision_free(r[i, :]):
                if verbosity > 2:
                    print("[Path Generator]: Path inside obstacle. Reducing integration step from {:5f} to {:5f}.".format(ds, ds*par['ds_decay_rate']))
                ds *= par['ds_decay_rate']
                r[i, :] = r[i - 1, :] + dr * ds
                # if ds < 0.01:
                #     import matplotlib.pyplot as plt
                #     plt.gca().quiver(*r[i-1, :], *dr, color='g')
                #     for o in obstacles:
                #         plt.gca().quiver(*o.boundary_mapping(r[i-1, :]), *o.normal(r[i-1, :]))
                #     plt.show()

                # Additional compute time check
                if toc(t0) > par['max_rhrp_compute_time']:
                    ri_collision = True
                    break
        if ri_collision:
            continue

        # Update travelled distance
        s[i] = s[i - 1] + ds
        # Try to increase step rate again
        ds = min(par['ds_increase_rate'] * ds, path_ds)
        # Increase iteration counter
        i += 1

    r = r[:i, :]
    s = s[:i]

    # Evenly spaced path
    path = np.vstack((np.interp(path_s, s, r[:, 0]), np.interp(path_s, s, r[:, 1]))).T

    compute_time = toc(t0)
    return path, s[-1], compute_time
