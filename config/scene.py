import numpy as np
from starworlds.obstacles import Ellipse, StarshapedPolygon, motion_model, Polygon, StarshapedPrimitiveCombination
from starworlds.utils.misc import draw_shapely_polygon
import matplotlib.pyplot as plt
import shapely


def zig_zag_motion_model(init_pos, p1, p2, vel):
    return motion_model.Waypoints(init_pos, [p1, p2] * 10, vel=vel)


def scene_config(id=None):
    scene_id = 0
    scene_description = {}
    obstacles_to_plot = None
    workspace = None
    ws_attractors = None
    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'No obstacle. Single attractor.'
    if id == scene_id:
        # workspace = StarshapedPolygon([[-5.3, -2.3], [5.3, -2.3], [5.3, 2.3], [-5.3, 2.3]])
        obstacles = [
        ]
        p0 = np.array([-5., 2.])
        reference_path = [[3, 0]]
        theta0 = 0 * np.pi / 2
        xlim = [-10, 10]
        ylim = [-10, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'No obstacle. Infinity path.'
    if id == scene_id:
        # workspace = StarshapedPolygon([[-6.3, -2.3], [6.3, -2.3], [6., 2.3], [-6., 2.3]])
        obstacles = [
        ]
        reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
                           3 * np.sin(4 * np.pi / 90 * s)]
                          for s in range(int(1.5 * 90))]
        p0 = np.array(reference_path[0])
        theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
                            reference_path[1][0] - reference_path[0][0])
        xlim = [-10, 10]
        ylim = [-10, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Single circle obstacle. Single attractor.'
    if id == scene_id:
        workspace = Ellipse([6., 6.], motion_model=motion_model.Static(pos=[0., 0.]))
        workspace = StarshapedPolygon([[-6., -6.], [6., -6.], [6., 6.], [-6., 6.]])
        obstacles = [
            Ellipse([2., 7.], motion_model=motion_model.Static(pos=[0., -2.]))
        ]
        reference_path = [[2.2, -5.2]]
        reference_path = [[5, -4.]]
        p0 = np.array([-5, 0])
        theta0 = 0
        xlim = [-10, 10]
        ylim = [-10, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Single circle obstacle. Infinity path.'
    if id == scene_id:
        obstacles = [
            Ellipse([1., 1.], motion_model=motion_model.Static(pos=[0., 0.]))
        ]
        reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
                           3 * np.sin(4 * np.pi / 90 * s)]
                          for s in range(int(1.5 * 90))]
        p0 = np.array(reference_path[0])
        theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
                            reference_path[1][0] - reference_path[0][0])
        xlim = [-10, 10]
        ylim = [-10, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Cross obstacle and one circle. Infinity path.'
    if id == scene_id:
        obstacles = [
            StarshapedPolygon([[-1.5, -0.2], [1.5, -0.2], [1.5, 0.2], [-1.5, 0.2]]),
            StarshapedPolygon([[-0.2, -1.5], [-0.2, 1.5], [0.2, 1.5], [0.2, -1.5]]),
            Ellipse([1., 1.], motion_model=motion_model.Static(pos=[4., 3.5]))
        ]
        reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
                           3 * np.sin(4 * np.pi / 90 * s)]
                          for s in range(90)]
        p0 = np.array(reference_path[0])
        theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
                            reference_path[1][0] - reference_path[0][0])
        xlim = [-10, 10]
        ylim = [-10, 10]
        obstacles_to_plot = [StarshapedPrimitiveCombination(obstacles[:2], None, xr=[0, 0]), obstacles[-1]]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Two circle obstacles. Infinity path.'
    if id == scene_id:
        obstacles = [
            Ellipse([1., 1.], motion_model=motion_model.Static(pos=[0., 0.])),
            Ellipse([1., 1.], motion_model=motion_model.Static(pos=[4., 3.5])),
        ]
        reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
                           3 * np.sin(4 * np.pi / 90 * s)]
                          for s in range(2 * 90)]
        p0 = np.array(reference_path[0])
        theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
                            reference_path[1][0] - reference_path[0][0])
        xlim = [-10, 10]
        ylim = [-10, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Static crowd. Single attractor.'
    if id == scene_id:
        workspace = StarshapedPolygon([[-3.5, -2], [5.5, -2], [5.5, 2], [-3.5, 2]])
        obstacles = [
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([-1, 1.7])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([-.5, 0.9])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([-1, 0.1])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([3, 1.6])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([4, 0.8])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([0.5, 2.8])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.3, 1.4])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1., -.4])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.3, -1.2])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([3., 2.2])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.3, -2.2]))
        ]
        reference_path = [[5., 1]]
        p0 = np.array([-3, 1])
        theta0 = 0
        xlim = [-4, 6]
        ylim = [-4, 6]


    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Static crowd. Infinity path.'
    if id == scene_id:
        obstacles = [
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([4.3, 3.2])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.2, 1.3])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.4, 2])),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.4, 0.5])),

            # Ellipse([0.7, 0.7], motion_model=motion_model.Static([4.3, 3.2])),
            # Ellipse([0.7, 0.7], motion_model=motion_model.Static([1.2, 1.3])),
            # Ellipse([0.7, 0.7], motion_model=motion_model.Static([1.5, 2.5])),
            # Ellipse([0.7, 0.7], motion_model=motion_model.Static([1.5, 0.])),
            StarshapedPolygon([[0, 0], [3, 0], [3, 1], [0, 1]], motion_model=motion_model.Static([-4, -2], rot=-np.pi/4), is_convex=True),
            StarshapedPolygon([[0, 0], [1, 0], [1, 3], [0, 3]], motion_model=motion_model.Static([-4, -2], rot=-np.pi/4), is_convex=True),

            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([-1, 1.7])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([-.5, 0.9])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([-1, 0.1])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([3, 1.6])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([4, 0.8])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([0.5, 2.8])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.3, 1.4])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([1., -.4])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.3, -1.2])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([3., 2.2])),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Static([1.3, -2.2]))
        ]
        obstacles_to_plot = obstacles[:-2] + [StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[-2:]]))]
        reference_path = [[6 * np.cos(2 * np.pi * th / 36.5),
                           3 * np.sin(4 * np.pi * th / 36.5)]
                          for th in np.linspace(0,36.5,100)]
        p0 = np.array(reference_path[0]) #+ np.array([-2, 0])
        theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
                            reference_path[1][0] - reference_path[0][0])
        xlim = [-6.5, 6.5]
        ylim = [-5, 5]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Dynamic crowd. Single attractor.'
    if id == scene_id:
        obstacles = [
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[6, -5], x1_mag=-0.2)),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[4, -3], x1_mag=-0.5)),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[7, -1], x1_mag=-0.2)),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[5, -2], x1_mag=-0.25, x2_mag=-0.2)),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[1, -5], x1_mag=0.3, x2_mag=0.5)),
            Ellipse([0.4, 1], motion_model=motion_model.SinusVelocity(pos=[8, -3], x2_mag=2, x2_period=5)),
            # Ellipse([2, 1], motion_model=motion_model.SinusVelocity(pos=[7, -2], x1_mag=2, x1_period=3)),
            # Ellipse([0.4, 0.4], motion_model=motion_model.SinusVelocity(pos=[8, -5], x1_mag=0.3, x2_mag=0.5)),
        ]
        # reference_path = [[0., -3.], [11, -3]]
        reference_path = [[11, -3]]
        p0 = np.array([0., -3.])
        theta0 = 0
        xlim = [-1, 12]
        ylim = [-8, 4]


    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Static mixed rectangle and circle scene. Single attractor.'
    if id == scene_id:
        ell_ax = [0.8, 0.8]
        obstacles = [
            StarshapedPolygon([(2, 6), (3, 6), (3, 10), (2, 10)], motion_model=motion_model.Static([2, -3]), is_convex=True),
            StarshapedPolygon([(3, 9), (3, 10), (6, 10), (6, 9)], motion_model=motion_model.Static([2, -3]), is_convex=True),
            StarshapedPolygon([(3, 6), (3, 7), (0, 7), (0, 6)], motion_model=motion_model.Static([2, -3]), is_convex=True),
            StarshapedPolygon([(8, 6), (9, 6), (9, 10), (8, 10)],  motion_model=motion_model.Static([-1, -6]), is_convex=True),
            StarshapedPolygon([(0, 1.2), (2.5, 1.2), (2.5, 2), (0, 2)], is_convex=True),
            StarshapedPolygon([(0, 0), (2.5, 0), (2.5, 0.8), (0, 0.8)], is_convex=True),
            StarshapedPolygon([(2, 0), (2.5, 0), (2.5, 2), (2, 2)], is_convex=True),
            Ellipse([0.5, 2], n_pol=100, motion_model=motion_model.Static([2, 9], rot=np.pi/4)),
            Ellipse([0.5, 2], n_pol=100, motion_model=motion_model.Static([2, 7], rot=-np.pi/4)),
            Ellipse(ell_ax, motion_model=motion_model.Static([0.5, 6.5])),
            Ellipse(ell_ax, motion_model=motion_model.Static([6.5, 7.5])),
            Ellipse(ell_ax, motion_model=motion_model.Static([6.5, 1])),
        ]
        reference_path = [[9., 5.]]
        p0 = np.array([1, 8.])
        theta0 = 0
        xlim = [-1, 12]
        ylim = [-1, 12]

        workspace = StarshapedPolygon([[-6, -6], [6, -6], [6, 6], [-6, 6]])
        obstacles = [
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([5, -1.5])),
            StarshapedPolygon([(0, 0), (1, 0), (1, 5), (0, 5)], motion_model=motion_model.Static([-2, -1]),
                              is_convex=True),
            Ellipse([0.5, 1.5], motion_model=motion_model.Static([-3, 3], rot=-np.pi/4)),
            Ellipse([0.5, 1.5], motion_model=motion_model.Static([-3, 0], rot=np.pi/4)),
            StarshapedPolygon([(0, 0), (1, 0), (1, 4), (0, 4)], motion_model=motion_model.Static([1, -6]),
                              is_convex=True),
            StarshapedPolygon([(0, 3), (4, 3), (4, 4), (0, 4)], motion_model=motion_model.Static([1, -6]),
                              is_convex=True),
            StarshapedPolygon([(0, 0), (1, 0), (1, 4), (0, 4)], motion_model=motion_model.Static([0, 1]),
                              is_convex=True),
            StarshapedPolygon([(3, 0), (4, 0), (4, 4), (3, 4)], motion_model=motion_model.Static([0, 1]),
                              is_convex=True),
            StarshapedPolygon([(0, 0), (4, 0), (4, 1), (0, 1)], motion_model=motion_model.Static([0, 1]),
                              is_convex=True),
            StarshapedPolygon([(1.5, 1), (1.5, -2), (2.5, -2), (2.5, 1)], motion_model=motion_model.Static([0, 1]),
                              is_convex=True),
            StarshapedPolygon([(1.5, -2), (4, -2), (4, -1), (1.5, -1)], motion_model=motion_model.Static([0, 1]),
                              is_convex=True),
            StarshapedPolygon([(0, 1.4), (3, 1.4), (3, 2.4), (0, 2.4)], motion_model=motion_model.Static([-4, -5]),
                              is_convex=True),
            StarshapedPolygon([(0, 0), (3, 0), (3, 1), (0, 1)], motion_model=motion_model.Static([-4, -5]),
                              is_convex=True),
            StarshapedPolygon([(0, 0), (1, 0), (1, 1.5), (0, 1.5)], motion_model=motion_model.Static([-2, -5]),
                              is_convex=True),
        ]
        xlim = [-7, 7]
        ylim = [-7, 7]
        reference_path = [[0., 0.]]
        p0 = np.array([-2.5, 1.5])
        # p0 = np.array([2.4, -5.6])
        p0 = np.array([3, .5])
        p0 = np.array([2, 2.5])
        # p0 = np.array([-5, 1.5])
        # p0 = np.array([-2.8, -3.8])
        obstacles_to_plot = obstacles[:4] + [Polygon(shapely.ops.unary_union([o.polygon() for o in obstacles[4:6]])),
                                             Polygon(shapely.ops.unary_union([o.polygon() for o in obstacles[6:11]])),
                                             Polygon(shapely.ops.unary_union([o.polygon() for o in obstacles[11:]]))]


    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Static mixed rectangle and circle scene. Given path.'
    if id == scene_id:
        ell_ax = [0.5, 0.5]
        obstacles = [
            StarshapedPolygon([(7, 5), (6, 5), (6, 3), (7, 3)], is_convex=True),
            StarshapedPolygon([(6, 3), (6, 7), (5, 7), (5, 3)], is_convex=True),
            StarshapedPolygon([(7, 1), (8, 1), (8, 5), (7, 5)], is_convex=True),
            StarshapedPolygon([(6, 7), (6, 8), (4, 8), (4, 7)], is_convex=True),
            StarshapedPolygon([(2, 6), (3, 6), (3, 10), (2, 10)], is_convex=True),
            StarshapedPolygon([(3, 9), (3, 10), (6, 10), (6, 9)], is_convex=True),
            StarshapedPolygon([(8, 6), (9, 6), (9, 10), (8, 10)], is_convex=True),
            Ellipse(ell_ax, motion_model=motion_model.Static([7.5, 8])),
            Ellipse(ell_ax, motion_model=motion_model.Static([4.8, 5])),
            Ellipse(ell_ax, motion_model=motion_model.Static([6.5, 1])),
            Ellipse(ell_ax, motion_model=motion_model.Static([3, 8.5])),
        ]
        reference_path = [[1., 9.], [1., 5.5], [3.5, 5.5], [3.5, 8.5], [7., 8.5], [7., 5.5], [9., 5.5], [9., 4]]
        p0 = np.array(reference_path[0])
        theta0 = -np.pi / 2
        xlim = [-1, 11]
        ylim = [-1, 11]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Static DSW equivalent scene. Single attractor.'
    if id == scene_id:
        workspace = StarshapedPolygon(shapely.ops.unary_union([
            shapely.geometry.Polygon([[-5, -6], [8, -6], [8, 1], [6, 1], [6, 6], [-5, 6], [-5, 0], [-3, -4]]),
            shapely.geometry.Point([2.5, -6]).buffer(2)]))
        # workspace = StarshapedPolygon([[-5, -6], [8, -6], [8, 1], [6, 1], [6, 6], [-5, 6], [-5, 0], [-3, -4]])
        # workspace = StarshapedPolygon([[-5, -8], [10, -8], [10, 6], [-5, 6]])

        obstacles = [
            StarshapedPolygon([[0, 0], [3, 0], [3, 1], [0, 1]], motion_model=motion_model.Static([-3, 1], np.pi/4), is_convex=True),
            StarshapedPolygon([[3, 0], [3, 3], [2, 3], [2, 0]], motion_model=motion_model.Static([-3, 1], np.pi/4), is_convex=True),
            StarshapedPolygon([[-4, 0], [3, 0], [3, 1], [-4, 1]], motion_model=motion_model.Static([3, -5]), is_convex=True),
            StarshapedPolygon([[0, 0], [0, 4], [-1, 4], [-1, 0]], motion_model=motion_model.Static([3, -5]), is_convex=True),
            StarshapedPolygon([[0, 0], [4, 0], [4, 1], [0, 1]], motion_model=motion_model.Static([-1.5, 0]), is_convex=True),
            StarshapedPolygon([[0, 0], [1, 0], [1, 4], [0, 4]], motion_model=motion_model.Static([0, -1.5]), is_convex=True),
            StarshapedPolygon([[-.5, 0], [2.5, 0], [2.5, 1], [-.5, 1]], motion_model=motion_model.Static([6, -3.5]), is_convex=True),
            StarshapedPolygon([[1.5, -2], [2.5, -2], [2.5, 3], [1.5, 3]], motion_model=motion_model.Static([6, -3.5]), is_convex=True),
            Ellipse([0.5, 0.5], motion_model=motion_model.Static([-1, 2.5])),
            Ellipse([0.8, 1.3], motion_model=motion_model.Static([4, 2], np.pi / 10)),
            Ellipse([0.8, 0.8], motion_model=motion_model.Static([3, 2.5])),
            Ellipse([1, 1], motion_model=motion_model.Static([1.5, -3.5])),
            Ellipse([0.8, 0.8], motion_model=motion_model.Static([-1.2, -1.8])),
            Ellipse([1, 1], motion_model=motion_model.Static([5, -1])),
            Ellipse([0.7, 0.7], motion_model=motion_model.Static([-2.8, -5.5])),
        ]
        reference_path = [[2.5, -7]]
        p0 = np.array([-3., 3])
        theta0 = 0
        xlim = [-6, 9]
        ylim = [-8, 7]
        obstacles_to_plot = [StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[:2]])),
                             StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[2:4]])),
                             StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[4:6]])),
                             StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[6:8]]))]\
                            + obstacles[8:]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Dynamic crowd. Single attractor.'
    if id == scene_id:
        mean_vel = 0.2
        obstacles = [
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[3, 5], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[2, 4], x2_mag=-(mean_vel+0.1))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[2, 9], x2_mag=-(mean_vel+0.3))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0, 6], x2_mag=-(mean_vel+0.1))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-4, 4], x2_mag=-(mean_vel+0.3))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[2, 2], x2_mag=-(mean_vel+0.1))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1.5, 3], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1.5, 3], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1.3, -1], x2_mag=-(mean_vel+0.3))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0, 0], x2_mag=-(mean_vel+0.1))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0.5, 4.5], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0.5, 7], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0.5, 10], x2_mag=-(mean_vel+0.1))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1, 11], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0.5, 2], x2_mag=-(mean_vel+0.4))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-0.5, 3], x2_mag=-(mean_vel+0.3))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1, 5], x2_mag=-(mean_vel+0.1))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-2.4, 7], x2_mag=-(mean_vel+0.2))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1.5, 8], x2_mag=-(mean_vel+0.3))),
            Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-3, 5], x2_mag=-(mean_vel+0.1))),
        ]
        # reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
        #                    3 * np.sin(4 * np.pi / 90 * s)]
        #                   for s in range(8 * 90)]
        # p0 = np.array(reference_path[0])
        # theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
        #                     reference_path[1][0] - reference_path[0][0])
        reference_path = [[0, 10.]]
        p0 = np.array([-4., -5.])
        # reference_path = [p0.tolist()] + reference_path
        theta0 = np.pi / 2
        xlim = [-8, 8]
        ylim = [-6, 12]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Corridor.'
    if id == scene_id:
        obstacles = [
            # StarshapedPolygon([[2, 2], [8, 2], [8, 3], [2, 3]]),
            StarshapedPolygon([[2, 5], [8, 5], [8, 6], [2, 6]]),
            StarshapedPolygon([[2, 2], [8, 2], [8, 3], [2, 3]]),
            StarshapedPolygon([[2, 8], [8, 8], [8, 9], [2, 9]]),
            Ellipse([1.1, 1.1], motion_model=motion_model.Interval([-2, 4], [(13, (10, 4))])),
            # StarshapedPolygon(Ellipse([1, 1]).polygon(), motion_model=motion_model.Interval([-1, 4], [(9, (10, 4))])),
            # Ellipse([1, 1], motion_model=motion_model.Interval([-2, 4], [(9, (11, 4))])),
        ]
        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        reference_path = [[0.5, 5.5]]
        theta0 = np.pi
        p0 = np.array([9, 4])
        xlim = [0, 10]
        ylim = [0, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Dynamic crowd and static polygons. Infinity path.'
    if id == scene_id:

        obstacles = [
            Ellipse([0.5, 0.5], motion_model=zig_zag_motion_model([4, 3], [4, 4], [4, -4], 0.2)),
            Ellipse([0.5, 0.5], motion_model=zig_zag_motion_model([3, 0], [3, 3], [3, -3], 0.2)),
            Ellipse([0.5, 0.5], motion_model=zig_zag_motion_model([-3, -3], [-3, -4], [-3, 4], 0.2)),
            Ellipse([0.5, 0.5], motion_model=zig_zag_motion_model([2.5, 0], [4, 0], [-4, 0], 0.2)),
            # Ellipse([0.5, 0.5], motion_model=motion_model.Waypoints(*zig_zag([4, 3], [4, 3], [4, -3]), vel=0.2)),
            # Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[2, 0], x2_mag=-.1, x2_period=30)),
            # Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-1, 0], x2_mag=.1, x2_period=20)),
            # Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[2, 0], x2_mag=-.3, x2_period=40)),
            # Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[-2, 0], x2_mag=.2, x2_period=30)),
            # Ellipse([0.5, 0.5], motion_model=motion_model.SinusVelocity(pos=[0, 0], x2_mag=.3, x2_period=25)),
            # StarshapedPolygon([[-3, -2], [-1, -2], [-1, -1.5], [-2.5, -1.5], [-2.5, 0], [-3, 0]])
            StarshapedPolygon([[-3, -2], [-1, -2], [-1, -1.5], [-3, -1.5]]),
            StarshapedPolygon([[-3, -2], [-2.5, -2], [-2.5, 0], [-3, 0]]),
        ]
        reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
                           3 * np.sin(4 * np.pi / 90 * s)]
                          for s in range(3 * 90)]
        p0 = np.array(reference_path[0])
        theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
                            reference_path[1][0] - reference_path[0][0])
        xlim = [-10, 10]
        ylim = [-10, 10]
        obstacles_to_plot = obstacles[:-2] + [Polygon(shapely.ops.unary_union([o.polygon() for o in obstacles[-2:]]))]


    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Ellipse trap. Single attractor.'
    if id == scene_id:
        obstacles = [
            Ellipse([1, .5], motion_model=motion_model.Static([5, 5], np.pi/2)),
            Ellipse([2, .5], motion_model=motion_model.Static([3, 6], 0)),
            Ellipse([2, .5], motion_model=motion_model.Static([3, 4], 0)),
            # StarshapedPolygon(Ellipse([1, 1]).polygon(), motion_model=motion_model.Interval([-1, 4], [(9, (10, 4))])),
            # Ellipse([1, 1], motion_model=motion_model.Interval([-2, 4], [(9, (11, 4))])),
        ]
        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        reference_path = [[9, 5]]
        theta0 = 0
        p0 = np.array([4, 5])
        xlim = [0, 10]
        ylim = [0, 10]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Several boundaries. Waypoints.'
    if id == scene_id:
        workspace = [
                     StarshapedPolygon([[-6, 3], [6, 3], [6, 6], [-6, 6]]),
                     StarshapedPolygon([[-6, -8], [-3, -8], [-3, 6], [-6, 6]]),
                     StarshapedPolygon([[-6, -8], [6, -8], [6, -5], [-6, -5]]),
                     StarshapedPolygon([[3, -8], [6, -8], [6, 0], [3, 0]]),
                     ]
        ws_attractors = [
            [-4.5, 4.5], #[[-6., 3], [-3, 3]],
            [-3, -6],
            [5, -6], [5, -2]]
        ws_attractors = [[-3.5, 4.5], [-3.5, -6], [5, -6], [5, -2]]
        obstacles = [
            # Ellipse([0.7, 0.7], motion_model=zig_zag_motion_model([-3.5, -2], [-3.5, -4], [-3.5, 1], 0.2)),
            Ellipse([0.7, 0.7], motion_model=motion_model.Waypoints([-3.5, -2], [[-3.5, -4], [-3.5, -0.5]], vel=0.2)),
            StarshapedPolygon([[-5, -2], [-4, -2], [-4, 1], [-5, 1]]),
            Ellipse([1, 1], motion_model=zig_zag_motion_model([0, -6], [-4, -6], [1, -6], 0.2)),
            Ellipse([1, 1], motion_model=zig_zag_motion_model([-4, 3.6], [4, 3.6], [-4, 3.6], 0.2)),
            Ellipse([.5, .5], motion_model=zig_zag_motion_model([-0.5, 4.5], [-4, 4.5], [3, 4.5], 0.2)),
            # Ellipse([.5, .5], motion_model=zig_zag_motion_model([3, -6], [-3, -10], [3, -6], 0.2)),
            # StarshapedPolygon([[-3, -2], [-2, -2], [-2, 1], [-3, 1]]),
        ]
        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        theta0 = np.pi
        p0 = np.array([5, 4.5])
        reference_path = [ws_attractors[0]]
        xlim = [-8, 8]
        ylim = [-9, 7]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Several boundaries. Path.'
    if id == scene_id:
        workspace = [
                     StarshapedPolygon([[-6, 3], [6, 3], [6, 6], [-6, 6]]),
                     StarshapedPolygon([[-6, -8], [-3, -8], [-3, 6], [-6, 6]]),
                     StarshapedPolygon([[-6, -8], [6, -8], [6, -5], [-6, -5]]),
                     StarshapedPolygon([[3, -8], [6, -8], [6, 0], [3, 0]]),
                     ]
        ws_attractors = [
            [-4.5, 4.5], #[[-6., 3], [-3, 3]],
            [-3, -6],
            [5, -6], [5, -2]]
        ws_attractors = [[-3.5, 4.5], [-3.5, -6], [5, -6], [5, -2]]
        obstacles = [
            # Ellipse([0.7, 0.7], motion_model=zig_zag_motion_model([-3.5, -2], [-3.5, -4], [-3.5, 1], 0.2)),
            Ellipse([0.7, 0.7], motion_model=motion_model.Waypoints([-3.5, -2], [[-3.5, -4], [-3.5, -0.5]], vel=0.2)),
            StarshapedPolygon([[-5, -2], [-4, -2], [-4, 1], [-5, 1]]),
            Ellipse([1, 1], motion_model=zig_zag_motion_model([0, -6], [-4, -6], [1, -6], 0.2)),
            Ellipse([1, 1], motion_model=zig_zag_motion_model([-4, 3.6], [4, 3.6], [-4, 3.6], 0.2)),
            Ellipse([.5, .5], motion_model=zig_zag_motion_model([-0.5, 4.5], [-4, 4.5], [3, 4.5], 0.2)),
            # Ellipse([.5, .5], motion_model=zig_zag_motion_model([3, -6], [-3, -10], [3, -6], 0.2)),
            # StarshapedPolygon([[-3, -2], [-2, -2], [-2, 1], [-3, 1]]),
        ]
        # obstacles_to_plot = obstacles[:-1]
        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        theta0 = np.pi
        p0 = np.array([5, 4.5])
        reference_path = [p0.tolist()] + ws_attractors
        xlim = [-8, 8]
        ylim = [-9, 7]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Enclosed corridor.'
    if id == scene_id:
        workspace = StarshapedPolygon([[1, 3], [9, 3], [9, 8], [1, 8]])
        obstacles = [
            # StarshapedPolygon([[2, 5], [8, 5], [8, 6], [2, 6]]),
            # StarshapedPolygon([[2, 2], [8, 2], [8, 3], [2, 3]]),
            # StarshapedPolygon([[2, 8], [8, 8], [8, 9], [2, 9]]),
            # Ellipse([1.1, 1.1], motion_model=motion_model.Interval([-2, 4], [(13, (10, 4))])),
            StarshapedPolygon([[3, 4], [8, 4], [8, 5.5], [3, 5.5]]),
            Ellipse([0.5, 0.5], n_pol=80, motion_model=motion_model.Interval([2, 3.5], [(20, (10, 3.5))])),
            Ellipse([0.5, 0.5], n_pol=80, motion_model=zig_zag_motion_model([6, 6.6], [6, 7.4], [6, 6.], 0.3)),
        ]

        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        reference_path = [[2, 5.5]]
        reference_path = [[2.5, 5]]
        theta0 = np.pi / 1.2
        p0 = np.array([7, 3.4])
        xlim = [0.5, 9.5]
        ylim = [2.5, 8.5]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'DSW example.'
    if id == scene_id:
        workspace = StarshapedPolygon([[0, 0], [17, 0], [17, 8], [11, 8], [11, 12], [5, 12], [5, 8], [0, 8]])
        obstacles = [
            # StarshapedPolygon([[2, 5], [8, 5], [8, 6], [2, 6]]),
            # StarshapedPolygon([[2, 2], [8, 2], [8, 3], [2, 3]]),
            # StarshapedPolygon([[2, 8], [8, 8], [8, 9], [2, 9]]),
            # Ellipse([1.1, 1.1], motion_model=motion_model.Interval([-2, 4], [(13, (10, 4))])),
            StarshapedPolygon([[7, 9], [12, 9], [12, 10], [7, 10]]),
            StarshapedPolygon([[13, 3], [14, 3], [14, 7], [11, 7], [11, 6], [13, 6]]),
            StarshapedPolygon([[4, -1], [8, -1], [8, 2], [6.5, 2], [6.5, 0.5], [4, 0.5]]),
            StarshapedPolygon([[12, -1], [15, -1], [15, 1], [12, 1]]),
            Ellipse([0.5, 1], motion_model=motion_model.Static([1.5, 2])),
            Ellipse([1, 1], motion_model=motion_model.Static([7, 7])),
            Ellipse([0.7, 1.2], motion_model=motion_model.Static([10, 1.5], np.pi/4)),
            Ellipse([0.7, 1.2], motion_model=motion_model.Static([10, 2.5], -np.pi/4)),
            StarshapedPolygon([[3, 2.5], [4, 2.5], [4, 6.5], [3, 6.5]]),
            StarshapedPolygon([[3, 4], [9, 4], [9, 5], [3, 5]]),
        ]
        obstacles_to_plot = (obstacles[:-4] +
                             [StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[-4:-2]]))] +
                             [StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[-2:]]))])

        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        reference_path = [[2, 5.5]]
        reference_path = [[9, 11]]
        theta0 = np.pi/2
        p0 = np.array([16, 2])
        # p0 = np.array([9, 8])
        xlim = [-1.5, 17.5]
        ylim = [-1.5, 12.5]


    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Illustration of ModEnv changing obstacles.'
    if id == scene_id:
        obstacles = [
            StarshapedPolygon([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [0, 1]]),
            StarshapedPolygon([[0, 0], [1, 0], [1, 3], [0, 3]]),
            Ellipse([0.5, 1], motion_model=motion_model.Static([3, -.5], 0 * np.pi / 2)),
            Ellipse([.7, .7], motion_model=motion_model.Static([0, 3.2]))
        ]
        obstacles_to_plot = [Polygon(shapely.ops.unary_union([o.polygon() for o in obstacles[:2]]))] + obstacles[2:]
        reference_path = [[1.5, 1.6]]
        theta0 = 0
        p0 = np.array([0, -1])
        xlim = [-1.5, 4]
        ylim = [-2, 4.5]


    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'Dynamic crowd 2. Single attractor.'
    if id == scene_id:
        mean_vel = 0.3
        obstacles = [
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[3, 5], x2_mag=-(mean_vel+0.2))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[2, 4], x2_mag=-(mean_vel+0.1))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[2, 9], x2_mag=-(mean_vel+0.3))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[0, 6], x2_mag=-(mean_vel+0.1))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-4, 4], x2_mag=-(mean_vel+0.3))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[0, 2], x2_mag=-(mean_vel+0.1))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-1.5, 3], x2_mag=-(mean_vel+0.2))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-1.5, 10], x2_mag=-(mean_vel+0.2))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-1.3, -1], x2_mag=-(mean_vel+0.3))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[2, 0], x2_mag=-(mean_vel+0.1))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[0.5, 9], x2_mag=-(mean_vel+0.1))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-3, 7], x2_mag=-(mean_vel+0.2))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-0.5, 12], x2_mag=-(mean_vel+0.1))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-4, 12], x2_mag=-(mean_vel+0.2))),
            Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[3, 13], x2_mag=-(mean_vel+0.1))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-1, 11], x2_mag=-(mean_vel+0.2))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[0.5, 2], x2_mag=-(mean_vel+0.4))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-0.5, 3], x2_mag=-(mean_vel+0.3))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-1, 5], x2_mag=-(mean_vel+0.1))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-2.4, 7], x2_mag=-(mean_vel+0.2))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-1.5, 8], x2_mag=-(mean_vel+0.3))),
            # Ellipse([1, 1], motion_model=motion_model.SinusVelocity(pos=[-3, 5], x2_mag=-(mean_vel+0.1))),
        ]
        # reference_path = [[6 * np.cos(2 * np.pi / 90 * s),
        #                    3 * np.sin(4 * np.pi / 90 * s)]
        #                   for s in range(8 * 90)]
        # p0 = np.array(reference_path[0])
        # theta0 = np.arctan2(reference_path[1][1] - reference_path[0][1],
        #                     reference_path[1][0] - reference_path[0][0])
        reference_path = [[0, 10.]]
        p0 = np.array([-4., -5.])
        # reference_path = [p0.tolist()] + reference_path
        theta0 = np.pi / 2
        xlim = [-8, 8]
        ylim = [-6, 12]

    # ----------------------------------- #
    scene_id += 1
    scene_description[scene_id] = 'MPC steps illustration.'
    if id == scene_id:
        workspace = StarshapedPolygon([[0, 0], [17, 0], [17, 8], [11, 8], [11, 12], [5, 12], [5, 8], [0, 8]])
        obstacles = [
            # StarshapedPolygon([[2, 5], [8, 5], [8, 6], [2, 6]]),
            # StarshapedPolygon([[2, 2], [8, 2], [8, 3], [2, 3]]),
            # StarshapedPolygon([[2, 8], [8, 8], [8, 9], [2, 9]]),
            # Ellipse([1.1, 1.1], motion_model=motion_model.Interval([-2, 4], [(13, (10, 4))])),
            # StarshapedPolygon([[7, 9], [12, 9], [12, 10], [7, 10]]),
            # StarshapedPolygon([[4, -1], [8, -1], [8, 2], [6.5, 2], [6.5, 0.5], [4, 0.5]]),
            # StarshapedPolygon([[12, -1], [15, -1], [15, 1], [12, 1]]),
            Ellipse([0.5, 1], motion_model=motion_model.Static([1.5, 2])),
            Ellipse([1, 1], motion_model=motion_model.Static([7, 7])),
            Ellipse([0.7, 1.2], motion_model=motion_model.Static([10, 1.5], np.pi / 4)),
            Ellipse([0.7, 1.2], motion_model=motion_model.Static([10, 2.5], -np.pi / 4)),
            StarshapedPolygon([[3, 2.5], [4, 2.5], [4, 6.5], [3, 6.5]]),
            StarshapedPolygon([[3, 4], [9, 4], [9, 5], [3, 5]]),
            StarshapedPolygon([[13, 3], [14, 3], [14, 7], [13, 7]]),
            StarshapedPolygon([[13, 7], [16, 7], [16, 6], [13, 6]]),
            StarshapedPolygon([[13, 4], [16, 4], [16, 3], [13, 3]])
        ]
        obstacles_to_plot = (obstacles[:-5] +
                             [StarshapedPolygon(shapely.ops.unary_union([o.polygon() for o in obstacles[-5:-3]]))] +
                             [Polygon(shapely.ops.unary_union([o.polygon() for o in obstacles[-3:]]))])

        # reference_path = [[9, 4.5], [0.5, 4.5], [0.5, 5.5]]
        # self.p0 = np.array(self.reference_path[0])
        reference_path = [[2, 5.5]]
        reference_path = [[9, 11]]
        theta0 = np.pi / 2 * 0.2
        p0 = np.array([14.5, 5])
        # p0 = np.array([9, 8])
        xlim = [-1.5, 17.5]
        ylim = [-1.5, 12.5]

    while id is None or not (1 <= id <= scene_id):
        print("Select scene ID\n -------------")
        for i, description in scene_description.items():
            print(str(i) + ": " + description)
        try:
            id = int(input("-------------\nScene ID: "))
        except:
            pass
        # if not valid_id(id):
        #     print("Invalid scene id: " + str(id) + ". Valid range [1, " + str(scene_id) + "].\n")

        return scene_config(id)

    if obstacles_to_plot is None:
        obstacles_to_plot = obstacles
    return p0, theta0, reference_path, ws_attractors, obstacles, workspace, obstacles_to_plot, xlim, ylim, id


#
# def scene_description():
#     return scene_config()

class Scene:

    def __init__(self, id=None, robot_radius=0.):
        self.id = id
        self.p0, self.theta0, self.reference_path, self.ws_attractors, self.obstacles, self.workspace, \
        self._obstacles_to_plot, self.xlim, self.ylim, self.id = scene_config(id)
        # Compute all polygon
        [o.polygon() for o in self.obstacles]
        [o.kernel() for o in self.obstacles]
        [o.is_convex() for o in self.obstacles]
        self.robot_radius = robot_radius
        # if robot_radius > 0:
        #     self.obstacles = [o.dilated_obstacle(padding=robot_radius, id="duplicate") for o in self.obstacles]
        self._static_obstacle = [o._motion_model is None or o._motion_model.__class__.__name__ == "Static" for o in self.obstacles]
        self.is_static = all(self._static_obstacle)

    def step(self, dt, robot_pos):
        if self.is_static:
            return
        for i, o in enumerate(self.obstacles):
            if self._static_obstacle[i]:
                continue
            prev_pos, prev_rot = o.pos().copy(), o.rot()
            o.move(dt)
            # if not o.exterior_point(robot_pos):
            # NOTE: Not well implemented. E.g. with hardcoded dt
            if o.polygon().distance(shapely.geometry.Point(robot_pos)) < max(abs(o._motion_model.lin_vel()))*0.2:
                print("[Scene]: Obstacle " + str(o) + " stands still to avoid moving into robot.")
                o._motion_model.set_pos(prev_pos)
                o._motion_model.set_rot(prev_rot)
        for i in range(len(self.obstacles)):
            if self._static_obstacle[i]:
                continue
            if self.obstacles[i]._motion_model is not None:
                self._obstacles_to_plot[i]._motion_model.set_pos(self.obstacles[i].pos())
                self._obstacles_to_plot[i]._motion_model.set_rot(self.obstacles[i].rot())

    def init_plot(self, ax=None, workspace_color='k', workspace_alpha=0.6, obstacle_color='lightgrey', obstacle_edge_color='k', obstacle_alpha=1, show_obs_name=False, draw_p0=1,
                  draw_ref=1, reference_color='y', reference_alpha=1, reference_marker='*', reference_markersize=14):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal')
        line_handles = []
        if draw_p0:
            ax.plot(*self.p0, 'ks', markersize=10)
        if draw_ref:
            if len(self.reference_path) > 1:
                ax.plot([r[0] for r in self.reference_path], [r[1] for r in self.reference_path], color=reference_color,
                             alpha=reference_alpha, zorder=-2)
            else:
                if isinstance(self.workspace, list):
                    [ax.plot(*attr, color=reference_color, alpha=reference_alpha,
                            marker=reference_marker,
                            markersize=reference_markersize) for attr in self.ws_attractors]
                else:
                    ax.plot(*self.reference_path[-1], color=reference_color, alpha=reference_alpha, marker=reference_marker,
                                 markersize=reference_markersize)

        if self.workspace is not None:
            draw_shapely_polygon(pol=shapely.geometry.box(self.xlim[0], self.ylim[0], self.xlim[1], self.ylim[1]), ax=ax, fc=obstacle_color, alpha=obstacle_alpha, ec='None', zorder=-10)
            if isinstance(self.workspace, list):
                [b.draw(ax=ax, fc='w', ec='None', show_reference=False, zorder=-9) for b in self.workspace]
            else:
                self.workspace.draw(ax=ax, fc='w', ec=obstacle_edge_color, show_reference=False, zorder=-9)

        for o in self._obstacles_to_plot:
            lh, _ = o.init_plot(ax=ax, fc=obstacle_color, alpha=obstacle_alpha, ec=obstacle_edge_color, show_name=show_obs_name, show_reference=False)
            line_handles.append(lh)

        return line_handles, ax

    def update_plot(self, line_handles):
        for i, o in enumerate(self._obstacles_to_plot):
            o.update_plot(line_handles[i])

    def draw(self, ax=None, workspace_color='k', workspace_alpha=0.6, obstacle_color='lightgrey', obstacle_edge_color='k', obstacle_alpha=1, show_obs_name=False, draw_p0=1,
                  draw_ref=1, reference_color='y', reference_alpha=1, reference_marker='*', reference_markersize=14):
        lh, ax = self.init_plot(ax, workspace_color, workspace_alpha, obstacle_color, obstacle_edge_color, obstacle_alpha, show_obs_name, draw_p0,
                  draw_ref, reference_color, reference_alpha, reference_marker, reference_markersize)
        self.update_plot(lh)
        return lh, ax
