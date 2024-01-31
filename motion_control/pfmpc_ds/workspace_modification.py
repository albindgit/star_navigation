import numpy as np
from starworlds.starshaped_hull import cluster_and_starify, ObstacleCluster
from starworlds.utils.misc import tic, toc
import shapely

def rho_environment(workspace, obstacles, rho):
    obstacles_rho = [o.dilated_obstacle(padding=rho, id="duplicate") for o in obstacles]
    workspace_rho = workspace.dilated_obstacle(padding=-rho, id="duplicate")
    obstacles_rho_sh = shapely.ops.unary_union([o.polygon() for o in obstacles_rho])
    free_rho_sh = workspace_rho.polygon().difference(obstacles_rho_sh)
    return obstacles_rho, workspace_rho, free_rho_sh, obstacles_rho_sh


def extract_r0(r_plus, p, rho, free_rho_sh, workspace_rho, obstacles_rho):
    initial_reference_set = shapely.geometry.Point(p).buffer(rho).intersection(free_rho_sh.buffer(-1e-2))
    if initial_reference_set.is_empty:
        if rho < 1e-5:
            print(rho, free_rho_sh.exterior.distance(shapely.geometry.Point(p)))
            import matplotlib.pyplot as plt
            from starworlds.utils.misc import draw_shapely_polygon
            _, ax = plt.subplots()
            draw_shapely_polygon(free_rho_sh.buffer(-1e-2), ax=ax, fc='g')
            ax.plot(*r_plus, 'rs')
            ax.plot(*p, 'ks')
            ax.plot(*shapely.geometry.Point(p).buffer(rho).exterior.xy, 'b--')
            for o in obstacles_rho:
                o.draw(ax=ax, fc='r', alpha=0.3)
                # ax.plot(*o.polygon().exterior.xy, 'k--')
            # for o in obstacles:
            #     o.draw(ax=ax, fc='lightgrey')
            plt.show()

        return None
    r0_sh, _ = shapely.ops.nearest_points(initial_reference_set, shapely.geometry.Point(r_plus))
    r0 = np.array(r0_sh.coords[0])
    if not all([o.exterior_point(r0) for o in obstacles_rho] + [workspace_rho.interior_point(r0)]):
        # initial_reference_set = initial_reference_set.buffer(-rho_buffer)
        r0_sh = initial_reference_set.centroid
        r0 = np.array(r0_sh.coords[0])
    if not all([o.exterior_point(r0) for o in obstacles_rho] + [workspace_rho.interior_point(r0)]):
        return None
    return r0

def workspace_modification(obstacles, workspace, p, pg, r_plus, rho0, hull_epsilon, previous_obstacle_clusters=None, params=None, verbosity=0):
    par = {'max_obs_compute_time': np.inf, 'gamma': 0.5, 'make_convex': 1, 'iterative_rho_reduction': 1,
          'use_previous_workspace': 1}
    for k in par.keys():
        if k in params:
            par[k] = params[k]

    p_sh = shapely.geometry.Point(p)
    # Clearance variable initialization
    t_init = tic()
    rho = rho0

    # Pad obstacles with rho
    obstacles_rho, workspace_rho, free_rho_sh, obstacles_rho_sh = rho_environment(workspace, obstacles, rho)
    r0 = extract_r0(r_plus, p, rho, free_rho_sh, workspace_rho, obstacles_rho)
    while r0 is None:
        if par['iterative_rho_reduction'] or rho < rho0:
            rho *= par['gamma']
        else:
            print(min([o.polygon().distance(p_sh) for o in obstacles] + [workspace_rho.polygon().exterior.distance(p_sh)]))
            obstacles_dist = min([o.polygon().distance(p_sh) for o in obstacles] + [workspace_rho.polygon().exterior.distance(p_sh)])
            rho = 0.8 * obstacles_dist
        obstacles_rho, workspace_rho, free_rho_sh, obstacles_rho_sh = rho_environment(workspace, obstacles, rho)
        r0 = extract_r0(r_plus, p, rho, free_rho_sh, workspace_rho, obstacles_rho)
        if verbosity > 1:
            print("[Workspace modification]: Reducing rho to " + str(rho))



        # r0 = extract_r0(initial_reference_set, r_plus, workspace_rho, obstacles_rho)

        # initial_reference_set = initial_reference_set.buffer(-0.2 * initial_reference_set.minimum_clearance)

        # Initial reference position selection
        # r0_sh, _ = shapely.ops.nearest_points(initial_reference_set, shapely.geometry.Point(r_plus))
        # r0 = np.array(r0_sh.coords[0])
        # if not all([o.exterior_point(r0) for o in obstacles_rho] + [workspace_rho.interior_point(r0)]):
        #     # initial_reference_set = initial_reference_set.buffer(-rho_buffer)
        #     r0_sh = initial_reference_set.centroid
        #     r0 = np.array(r0_sh.coords[0])
        #
        # if not all([o.exterior_point(r0) for o in obstacles_rho] + [workspace_rho.interior_point(r0)]):
        #     if verbosity > 1:
        #         print("[Workspace Modification]: r0 not valid. Reducing rho to " + str(rho * par['gamma']))



    # Goal reference position selection
    rg_sh = shapely.geometry.Point(pg)
    rg = pg
    rho_buffer = 0
    while not all([o.exterior_point(rg) for o in obstacles_rho] + [workspace_rho.interior_point(rg)]):
        rg_sh, _ = shapely.ops.nearest_points(workspace_rho.polygon().buffer(-rho_buffer).difference(obstacles_rho_sh.buffer(rho_buffer)), shapely.geometry.Point(pg))
    # while not all([o.exterior_point(rg) for o in obstacles_rho]):
    #     rg_sh, _ = shapely.ops.nearest_points(shapely.geometry.box(-1e6,-1e6,1e6,1e6).difference(obstacles_rho_sh.buffer(rho_buffer)), shapely.geometry.Point(pg))
        rg = np.array(rg_sh.coords[0])
        rho_buffer += 1e-2

    # TODO: Check more thoroughly
    if par['use_previous_workspace'] and previous_obstacle_clusters is not None:
        obstacles_star_sh = shapely.ops.unary_union([o.polygon() for o in previous_obstacle_clusters]).buffer(1e-4)
        if obstacles_star_sh.disjoint(shapely.geometry.Point(r0)) and obstacles_star_sh.disjoint(rg_sh) and obstacles_rho_sh.within(obstacles_star_sh):
            if verbosity > 2:
                print("[Workspace modification]: Reuse workspace from previous time step.")
            obstacle_clusters = previous_obstacle_clusters
            exit_flag = 10
            compute_time = toc(t_init)
            obstacle_timing = None
            return obstacle_clusters, r0, rg, rho, obstacles_rho, workspace_rho, compute_time, obstacle_timing, exit_flag
        else:
            if verbosity > 2:
                print("[Workspace modification]: No reuse workspace from previous time step.")
                print(obstacles_star_sh.disjoint(shapely.geometry.Point(r0)), obstacles_star_sh.disjoint(rg_sh), obstacles_rho_sh.within(obstacles_star_sh))
    else:
        if par['use_previous_workspace'] and verbosity > 2:
            print("[Workspace modification]: No reuse workspace from previous time step.")
                # import matplotlib.pyplot as plt
                # from starworlds.utils.misc import draw_shapely_polygon
                # _,ax = plt.subplots()
                # ax.plot([obstacles_rho_sh.bounds[0], obstacles_rho_sh.bounds[2]],
                #         [obstacles_rho_sh.bounds[1], obstacles_rho_sh.bounds[3]], alpha=0)
                # draw_shapely_polygon(obstacles_star_sh, ax=ax, fc='g', alpha=0.7)
                # draw_shapely_polygon(obstacles_rho_sh.difference(obstacles_star_sh), ax=ax, fc='r', alpha=1, ec='k', linestyle='--')
                # plt.show()


    # Apply cluster and starify
    obstacle_clusters, obstacle_timing, exit_flag, n_iter = cluster_and_starify(obstacles_rho, r0, rg, hull_epsilon,
                                                                                workspace=workspace_rho,
                                                                                max_compute_time=par['max_obs_compute_time']-toc(t_init),
                                                                                previous_clusters=previous_obstacle_clusters,
                                                                                make_convex=par['make_convex'], verbose=verbosity,
                                                                                timing_verbose=0)

    compute_time = toc(t_init)
    return obstacle_clusters, r0, rg, rho, obstacles_rho, workspace_rho, compute_time, obstacle_timing, exit_flag