import numpy as np
from robot import MobileRobot


class Omnidirectional(MobileRobot):

    def __init__(self, radius, vel_max, name='robot'):
        vel_max = vel_max * np.ones(2)
        self.vmax = float(np.linalg.norm(vel_max))
        super().__init__(nu=2, nx=2, radius=radius, name=name, u_min=(-vel_max).tolist(), u_max=(vel_max).tolist())

    def f(self, x, u):
        return [u[0],
                u[1]]

    def h(self, x):
        return x[:2]

    def vel_min(self):
        return self.u_min

    def vel_max(self):
        return self.u_max
