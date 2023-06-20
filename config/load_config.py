import yaml
import pathlib
from config.scene import Scene
from motion_control.soads import SoadsController
from motion_control.dmp import DMPController
from motion_control.pfmpc_ds import MotionController as PFMPC_ds
from motion_control.pfmpc_obstacle_constraints import MotionController as PFMPC_obstacle_constraints
from motion_control.pfmpc_artificial_reference import MotionController as PFMPC_artificial_reference
from robot import Unicycle, Omnidirectional, Bicycle
import numpy as np


def load_config(scene_id=None, robot_type_id=None, ctrl_param_file=None, verbosity=0):
    param_path = pathlib.PurePath(__file__).parents[0].joinpath('params')


    # Load robot
    robot_param_path = str(param_path.joinpath('robot_params.yaml'))
    with open(r'' + robot_param_path) as stream:
        params = yaml.safe_load(stream)
    if robot_type_id is None:
        print("Select robot ID\n -------------")
        for i, k in enumerate(params.keys()):
            print(str(i) + ": " + k)
        robot_type_id = int(input("-------------\nRobot type: "))
    robot_type = list(params.keys())[robot_type_id]
    robot_params = params[robot_type]

    scene = Scene(scene_id, robot_radius=robot_params['radius'])

    if robot_params['model'] == 'Omnidirectional':
        robot = Omnidirectional(radius=robot_params['radius'],
                                vel_max=robot_params['vel_max'],
                                name=robot_type)
        x0 = scene.p0
    elif robot_params['model'] == 'Unicycle':
        robot = Unicycle(radius=robot_params['radius'], vel_min=[robot_params['lin_vel_min'], -robot_params['ang_vel_max']],
                         vel_max=[robot_params['lin_vel_max'], robot_params['ang_vel_max']],
                         name=robot_type)
        try:
            x0 = np.append(scene.p0, [scene.theta0])
        except AttributeError:
            x0 = np.append(scene.p0, [np.arctan2(scene.pg[1]-scene.p0[1], scene.pg[0]-scene.p0[0])])
    elif robot_params['model'] == 'Bicycle':
        robot = Bicycle(radius=robot_params['radius'],
                        vel_min=[robot_params['lin_vel_min'], -robot_params['steer_vel_max']],
                        vel_max=[robot_params['lin_vel_max'], robot_params['steer_vel_max']],
                        steer_max=robot_params['steer_max'],
                        name=robot_type)
        try:
            x0 = np.append(scene.p0, [scene.theta0, 0])
            # x0 = np.append(scene.p0, [scene.theta0])
        except AttributeError:
            x0 = np.append(scene.p0, [np.arctan2(scene.pg[1]-scene.p0[1], scene.pg[0]-scene.p0[0]), 0])
            # x0 = np.append(scene.p0, [np.arctan2(scene.pg[1]-scene.p0[1], scene.pg[0]-scene.p0[0])])
    else:
        raise Exception("[Load Config]: Invalid robot model.\n"
                        "\t\t\tSelection: {}\n"
                        "\t\t\tValid selections: [Omnidirectional, Unicycle, Bicycle]".format(robot_params['model']))

    # Load control parameters
    ctrl_param_path = str(param_path.joinpath(ctrl_param_file))
    # param_file = str(pathlib.PurePath(pathlib.Path(__file__).parent, ctrl_param_file))
    with open(r'' + ctrl_param_path) as stream:
        params = yaml.safe_load(stream)
    if 'soads' in params:
        controller = SoadsController(params['soads'], robot, verbosity)
    elif 'dmp' in params:
        controller = DMPController(params['dmp'], robot, scene.reference_path, verbosity)
    elif 'pfmpc_ds' in params:
        # if global_path is None:
        #     global_path = [scene.p0.tolist(), scene.pg.tolist()]
        controller = PFMPC_ds(params['pfmpc_ds'], robot, scene.reference_path, verbosity)
    elif 'pfmpc_obstacle_constraints' in params:
        controller = PFMPC_obstacle_constraints(params['pfmpc_obstacle_constraints'], robot, scene.reference_path, verbosity)
    elif 'pfmpc_artificial_reference' in params:
        controller = PFMPC_artificial_reference(params['pfmpc_artificial_reference'], robot, scene.reference_path, verbosity)
    else:
        raise Exception("[Load Config]: No valid controller selection in param file.\n"
                        "\t\t\tSelection: {}\n"
                        "\t\t\tValid selections: [soads, tunnel_mpc, pfmpc_ds]".format(str(list(params.keys())[0])))

    return scene, robot, controller, x0
