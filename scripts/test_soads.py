import numpy as np
import matplotlib.pyplot as plt
from config.load_config import load_config
from motion_control.soads import draw_vector_field, compute_weights

ctrl_param_file = 'soads_ctrl_params.yaml'
scene, robot, controller, x0 = load_config(ctrl_param_file=ctrl_param_file, robot_type_id=0, verbosity=3)

# Initialize
T_max = 30
dt = controller.params['dt']
K = int(T_max / dt)
u = np.zeros((robot.nu, K))
x = np.zeros((x0.size, K+1))
x[:, 0] = x0
pg = scene.reference_path[-1]
timing_history = {'obstacle': [], 'control': []}
previous_path = []
convergence_threshold = 0.01
converged = False
paused = True
step_once = False
i = 0

# Init plotting
fig_scene, ax_scene = plt.subplots()
ax_scene.set_xlabel('x1 [m]', fontsize=15)
ax_scene.set_ylabel('x2 [m]', fontsize=15)
travelled_path_handle = ax_scene.plot([], [], 'k-', linewidth=2)[0]
goal_handle = ax_scene.plot(*pg, 'g*', markersize=16)[0]
obstacle_handles, _ = scene.init_plot(ax=ax_scene, draw_p0=0, draw_ref=0, show_obs_name=1)
scene.update_plot(obstacle_handles)
robot.width = 0
robot_handles, _ = robot.init_plot(ax=ax_scene, color='y', alpha=1, markersize=16)
robot.update_plot(x0, robot_handles)
obstacle_star_handles = []
color_list = plt.cm.gist_ncar(np.linspace(0, 1, len(scene.obstacles)))

streamplot_handle = None

fig_open = True

def on_close(event):
    global fig_open
    fig_open = False

def on_press(event):
    global paused, step_once, streamplot_handle
    if event.key == ' ':
        paused = not paused
    elif event.key == 'right':
        step_once = True
        paused = True
    elif event.key == 't':
        fig_timing, ax = plt.subplots()
        ax.plot(timing_history['obstacle'], '-o', label='obstacle')
        ax.plot(timing_history['control'], '-o', label='control')
        fig_timing.canvas.draw()
    elif event.key == 'w':
        ax_scene.axis('off')
        ax_scene.title.set_visible(False)
        file_name = input("-------------\nFile name: ")
        fig_scene.savefig("utils/" + file_name, transparent=True)
        ax_scene.axis('on')
        ax_scene.title.set_visible(True)
    elif event.key == 'a':
        n = int(input("-------------\nStreamplot resolution: "))
        streamplot_handle = draw_vector_field(pg, controller.obstacles_star, ax_scene, workspace=scene.workspace, n=n, color='orange')
    else:
        print(event.key)

def remove_local_plots():
    global streamplot_handle
    # Remove streamplot
    if streamplot_handle is not None:
        from matplotlib.patches import FancyArrowPatch
        streamplot_handle.lines.remove()
        for art in ax_scene.get_children():
            if not isinstance(art, FancyArrowPatch):
                continue
            art.remove()  # Method 1
        streamplot_handle = None

fig_scene.canvas.mpl_connect('close_event', on_close)
fig_scene.canvas.mpl_connect('key_press_event', on_press)

ls = []

while fig_open and not converged:

    if i < K and (not paused or step_once):
        p = robot.h(x[:, i])
        step_once = False
        # Move obstacles
        scene.step(dt, p)
        # Compute mpc
        u[:, i] = controller.compute_u(x[:, i], pg, scene.obstacles, workspace=scene.workspace)
        # Integrate robot state with new control signal
        x[:, i+1], _ = robot.move(x[:, i], u[:, i], dt)
        # Add timing to history
        for k in timing_history.keys():
            timing_history[k] += [controller.timing[k]]

        # Update plots
        robot.update_plot(x[:, i], robot_handles)
        scene.update_plot(obstacle_handles)
        [h.remove() for h in obstacle_star_handles if h is not None]
        obstacle_star_handles = []
        for j, cl in enumerate(controller.obstacle_clusters):
            for o in cl.obstacles:
                lh, _ = o.draw(ax=ax_scene, fc='lightgrey', show_reference=False)
                obstacle_star_handles += lh
        for o in controller.obstacles_star:
            lh, _ = o.draw(ax=ax_scene, fc='red', show_reference=False, alpha=0.9, zorder=0)
            obstacle_star_handles += lh
            lh = ax_scene.plot(*o.xr(), 'gd', markersize=8)
            obstacle_star_handles += lh

        # obstacle_star_handles += [ax_scene.quiver(*scene.boundary.boundary_mapping(x[:, i]), *scene.boundary.normal(x[:, i]))]

        gamma = [obs.distance_function(p) for obs in controller.obstacles_star]
        w = compute_weights(gamma)

        if ls:
            [l.remove() for l in ls]
        ls = []
        for j, o in enumerate(controller.obstacles_star):
            ls += [ax_scene.text(*o.xr(), "{:.2f}".format(w[j]))]
            if o.reference_direction(p).dot(pg-p) > 0 and o.normal(p).dot(pg-p) > 0:
                ls += ax_scene.plot(*o.xr(), 'r+', ms=12, zorder=10)
            else:
                ls += ax_scene.plot(*o.xr(), 'k+', ms=12, zorder=10)
        #     b = o.boundary_mapping(x[:, i])
        #     dx_o = f(x[:, i], pg, [o])
        #     ls += ax_scene.plot(*b, 'yx')
        #     ls += [ax_scene.quiver(*b, *dx_o, zorder=3, color='y')]

        travelled_path_handle.set_data(x[0, :i+1], x[1, :i+1])

        ax_scene.set_title("Time: {:.1f} s".format(i*dt))

        i += 1

        fig_scene.canvas.draw()

    converged = np.linalg.norm(robot.h(x[:, i])-pg) < convergence_threshold

    if i == K or converged:
        ax_scene.set_title("Time: {:.1f} s. Finished".format(i * dt))
        fig_scene.canvas.draw()

    plt.pause(0.005)


ot = timing_history['obstacle']
print("Timing\n-----\nMean: {:.2f}\nMax: {:.2f}\nStdDev: {:.2f}".format(np.mean(ot), np.max(ot), np.std(ot)))
plt.figure()
plt.plot(ot)

# Wait until figure closed when converged
plt.show()
