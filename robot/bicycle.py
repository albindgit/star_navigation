import numpy as np
from robot import MobileRobot


class Bicycle(MobileRobot):

    def __init__(self, width, vel_min=None, vel_max=None, steer_max=None, name='robot'):
        self.vmax = vel_max[0]
        x_min = [-np.inf] * 3 + [-steer_max]
        x_max = [np.inf] * 3 + [steer_max]
        super().__init__(nu=2, nx=4, width=width, name=name, u_min=vel_min, u_max=vel_max, x_min=x_min, x_max=x_max)

    # [x,y of backwheel, orientation, steering angle]
    # def f(self, x, u):
    #     return [u[0] * np.cos(x[2]),
    #             u[0] * np.sin(x[2]),
    #             u[0] * np.tan(u[1]) / self.width]
    def f(self, x, u):
        return [u[0] * np.cos(x[2]),
                u[0] * np.sin(x[2]),
                u[0] * np.tan(u[1]) / self.width,
                u[1]]
    # def f(self, x, u):
    #     return [u[0] * np.cos(x[2]) * np.cos(x[3]),
    #             u[0] * np.sin(x[2]) * np.cos(x[3]),
    #             u[0] * np.sin(x[3]) / self.width,
    #             u[1]]

    def h(self, x):
        return [x[0] + self.width/2 * np.cos(x[2]),
                x[1] + self.width/2 * np.sin(x[2])]

    def vel_min(self):
        return self.u_min

    def vel_max(self):
        return self.u_max

    def init_plot(self, ax=None, color='b', alpha=0.7, markersize=10, **kwargs):
        h = super(Bicycle, self).init_plot(ax=ax, color=color, alpha=alpha)
        h += ax.plot(0, 0, marker=(3, 0, np.rad2deg(0)), markersize=markersize, color=color)
        h += ax.plot(0, 0, marker=(2, 0, np.rad2deg(0)), markersize=0.5*markersize, color='w')
        h += ax.plot(0, 0, color=color, alpha=alpha)
        return h

    def update_plot(self, x, handles):
        super(Bicycle, self).update_plot(x, handles)
        handles[1].set_data(self.h(x))
        handles[1].set_marker((3, 0, np.rad2deg(x[2]-np.pi/2)))
        handles[1].set_markersize(handles[1].get_markersize())
        handles[2].set_data(x[0], x[1])
        handles[2].set_marker((2, 0, np.rad2deg(x[2]-np.pi/2)))
        handles[2].set_markersize(handles[2].get_markersize())
        handles[3].set_data([x[0], x[0] + self.width * np.cos(x[2])],
                            [x[1], x[1] + self.width * np.sin(x[2])])