import numpy as np
import matplotlib.pyplot as plt
from robot.unicycle import Unicycle
import shapely

dr = 0.9
drth = np.pi/4
dp = 0.8
dpth = -np.pi/2
r0 = np.array([0, 0])
rho = 1
rg = r0 + dr * np.array([np.cos(drth), np.sin(drth)])
p = r0 + dp * np.array([np.cos(dpth), np.sin(dpth)])
p = np.array([-0.4, -0.4])
rg = np.array([0.1, 0.])
p = np.array([-0.4, -0.])
th = 0 + 1*np.pi

br0 = shapely.geometry.Point(r0).buffer(rho)
brg = shapely.geometry.Point(rg).buffer(rho)
intersection = br0.exterior.intersection(brg.exterior)
ip1 = np.array(intersection.geoms[0].coords[0])
ip2 = np.array(intersection.geoms[1].coords[0])

u_min = np.tile(-10, 2)
u_max = np.tile(10, 2)
u_min = [-1, -1]
u_max = [2, 1]


robot = Unicycle(0.1, vel_min=u_min, vel_max=u_max)

fig = plt.figure(1)
ax = fig.subplots()
ax.plot(*r0, 'ko')
ax.plot(*rg, 'gd')
an = np.linspace(0, 2*np.pi, 100)
ax.plot(r0[0] + rho * np.cos(an),r0[1] + rho * np.sin(an), 'k--')
ax.plot(rg[0] + rho * np.cos(an),rg[1] + rho * np.sin(an), 'g--')
# ax.plot([r[0], rg[0]], [r[1], rg[1]])
ax.plot(*zip(r0, rg), 'k-.')
ax.set_aspect('equal', 'box')

r_handle = ax.plot([], [], 'y*')[0]
r_ball_handle = ax.plot([], [], 'y--')[0]
r_line_handle = ax.plot([], [], 'k-.')[0]

u_hist_horizon = 20
ax_u = plt.figure().subplots(2,1)
u0_handle = ax_u[0].plot(np.arange(-u_hist_horizon, 0, 1), [None] * u_hist_horizon, '-o')[0]
ax_u[0].plot([-u_hist_horizon, 0], [u_min[0], u_min[0]], 'r--')
ax_u[0].plot([-u_hist_horizon, 0], [u_max[0], u_max[0]], 'r--')
ax_u[1].plot([-u_hist_horizon, 0], [u_min[1], u_min[1]], 'r--')
ax_u[1].plot([-u_hist_horizon, 0], [u_max[1], u_max[1]], 'r--')
u1_handle = ax_u[1].plot(np.arange(-u_hist_horizon, 0, 1), [None] * u_hist_horizon, '-o')[0]
ax_u[0].set_xlim([-u_hist_horizon, 0])
ax_u[0].set_ylim([u_min[0] - 0.1, u_max[0] + 0.1])
ax_u[1].set_xlim([-u_hist_horizon, 0])
ax_u[1].set_ylim([u_min[1] - 0.1, u_max[1] + 0.1])
u_history = [None] * 2 * u_hist_horizon

handle, _ = robot.init_plot(ax=ax)


x = np.array([p[0], p[1], th])



def kappa(r0, rg, x, t):
    k1, k2 = 0.5, 0.3
    # k1, k2 = 1, 1
    # print("k1_max: {:.2f}.".format(min(u_max[0], -u_min[0]) / (2*rho)))
    # print("k2_max: {:.2f}.".format(u_max[1] / np.pi))
    # p = x[:2]
    th = x[2]
    e0 = x[:2] - r0
    eg = x[:2] - rg

    if np.linalg.norm(eg) < rho:
        r = rg
    else:



        # tau = 1
        # r = tau * rg + (1 - tau) * r0
        # i = 0
        # while np.linalg.norm(r - x[:2]) - np.linalg.norm(r - ip1) > 0:
        #     # print(tau, np.linalg.norm(r - x[:2]), np.linalg.norm(r - ip1))
        #     i += 1
        #     tau_scale = 0.98
        #     tau *= tau_scale
        #     r = tau * rg + (1 - tau) * r0
        #     if i > 100:
        #         print("IT", tau)
        #         break

        p_scale = 1.1

        a = r0[1]-rg[1]
        b = rg[0]-r0[0]
        c = - (r0[1]*b + r0[0]*a)
        az = ip1[0]-x[0]
        bz = ip1[1]-x[1]
        cz = 0.5 * (x[:2].dot(x[:2]) - ip1.dot(ip1))
        # cz = 0.5 * (p_scale*x[:2].dot(x[:2]) - ip1.dot(ip1) - (1-p_scale)*x[:2].dot(ip1))

        # rx = (-cz*b+c*bz)/(-a*bz+az*b)
        # ry = -a/b * rx - c/b
        # r = np.array([rx, ry])
        r = np.array([(b*cz-bz*c)/(a*bz-az*b), (c*az-cz*a)/(a*bz-az*b)])
        # print(r)
        # print(np.linalg.norm(r - x[:2]) - np.linalg.norm(r - ip1))



        # tau /= tau_scale
        # r = tau * rg + (1 - tau) * r0
        # print(tau, np.linalg.norm(r - x[:2]), np.linalg.norm(r - ip1))
        # print(tau, np.linalg.norm(r - x[:2]), np.linalg.norm(r - ip))
    #
    #
    # r = np.array([r[0], r[1], 0])

    # c1,c2 = circles_from_p1p2r(ip1, ip2, np.linalg.norm(ip1-ip2)/2)
    # print(c1, c2)
    # print(0.5 * rg + 0.5 * r0)
    # k1, k2, k3 = 1, 0.7, 0.3
    #
    # def phi(i, s):
    #     return k[i] * np.tanh(s)
    #
    # r = np.zeros(3)
    # xe = x - r
    #
    # x_transformed = np.array([xe[0]*np.sin(xe[2]) - xe[1]*np.cos(xe[2]),
    #                           xe[0]*np.cos(xe[2]) + xe[1]*np.sin(xe[2]),
    #                           xe[2]])
    #
    # u = np.array([
    #     -k1 * np.tanh(xe[1]),
    #     -k2 * np.tanh(xe[2]) + k3 * np.tanh(xe[0]) * np.sin(t)
    # ])
    # return u, r

    e = x[:2] - r

    # Theta ref in (-pi, pi]
    r_th = np.arctan2(e[1], e[0]) + np.pi
    # while r_th > np.pi:
    #     r_th -= 2 * np.pi
    # while r_th <= -np.pi:
    #     r_th += 2 * np.pi

    e_th = r_th - th
    e_th0 = e_th
    while e_th > np.pi:
        e_th -= 2 * np.pi
    while e_th <= -np.pi:
        e_th += 2 * np.pi


    print("r: {:.2f}, th: {:.2f}, e0: {:.2f}, e: {:.2f}".format(*np.array([r_th, th, e_th0, e_th]) / np.pi))

    u = np.array([
        -k1 * (e[0]*np.cos(th) + e[1]*np.sin(th)),
        k2 * (e_th)
        ])

    if robot.u_min[0] == 0 and u[0] < 0:
        u[0] = 0

    u0_sig = u[0] / robot.u_max[0] if u[0] >= 0 else u[0] / robot.u_min[0]
    u1_sig = u[1] / robot.u_max[1] if u[1] >= 0 else u[1] / robot.u_min[1]

    sig = max(u0_sig, u1_sig, 1)

    if sig > 1:
        print("Saturation. " + str(sig))
        if u0_sig > u1_sig:
            u[0] = robot.u_max[0] if u[0] > 0 else robot.u_min[0]
            u[1] /= sig
        else:
            u[0] /= sig
            u[1] = robot.u_max[1] if u[1] > 0 else robot.u_min[1]

    return u, r

t = 0
dt = 0.1
T = 10
e_rg = np.linalg.norm(rg-x[:2])
robot.update_plot(x, handle)
while plt.fignum_exists(1):
    plt.pause(0.001)
    fig.waitforbuttonpress()

    u, r = kappa(r0, rg, x, t)
    u_history += u.tolist()

    robot.update_plot(x, handle)

    rho_r = np.linalg.norm(r[:2] - x[:2])

    # print(np.linalg.norm(r[:2] - x[:2])- np.linalg.norm(r[:2] - ip1))
    # def lz(t):
    #     z = .5 * (x[:2] + ip1)
    #     nz = np.array([x[1]-ip1[1], ip1[0]-x[0]])
    #     az = ip1[0] - x[0]
    #     bz = ip1[1] - x[1]
    #     cz = 0.5 * (x[:2].dot(x[:2]) - ip1.dot(ip1))
    #     z = np.array([0, -cz/bz])
    #     nz = np.array([1, -az/bz])
    #     return z + nz*t

    # r = 0.5 * rg + 0.5 * r0
    # rho_r = np.linalg.norm(r-ip1)
    # rl1, rl2 = lz(-100), lz(100)
    # r_line_handle.set_data([rl1[0], rl2[0]], [rl1[1], rl2[1]])
    r_handle.set_data(*r[:2])
    r_ball_handle.set_data(r[0] + rho_r * np.cos(an),r[1] + rho_r * np.sin(an))
    u0_handle.set_ydata(u_history[-2*u_hist_horizon::2])
    u1_handle.set_ydata(u_history[-2*u_hist_horizon+1::2])

    e_r0 = np.linalg.norm(r0-x[:2])
    e_rg_prev = e_rg
    e_rg = np.linalg.norm(rg-x[:2])
    if e_rg_prev < rho and e_rg > e_rg_prev:
        print("Increased error.")
        print(e_rg_prev, e_rg)

    if e_rg > rho and e_r0 > rho:
        print("Outside allowed region")
        plt.show( )

    x, _ = robot.move(x, u, dt)
    t += dt



plt.close('all')
