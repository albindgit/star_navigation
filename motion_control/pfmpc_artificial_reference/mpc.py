import casadi.casadi as cs
import opengen as og
import numpy as np
import os, sys
import yaml


class NoSolutionError(Exception):
    '''raise this when there's no solution to the mpc problem'''
    pass

def pol_eval(pol_coeff, x, n_pol):
    return sum([pol_coeff[i] * x ** (n_pol - i) for i in range(n_pol+1)])


def pol2pos(pol_coeff, s, n_pol):
    return [pol_eval(pol_coeff[:n_pol+1], s, n_pol), pol_eval(pol_coeff[n_pol+1:], s, n_pol)]

# Returns >0 for interior points, <0 for exterior points, =0 for boundary points
def ellipse_function(p, pos, rot, ax):
    x_dist = p[0] - pos[0]
    y_dist = p[1] - pos[1]
    s, c = cs.sin(rot), cs.cos(rot)
    normalized_center_dist = ((x_dist * c + y_dist * s) ** 2) / (ax[0] ** 2) + ((x_dist * s + y_dist * c) ** 2) / (ax[1] ** 2)
    return 1 - normalized_center_dist

# Returns >0 for interior points, otherwise 0
def convex_polygon_function(p, vertices, N):
    hp_vals = []
    for i in range(N-1):
        idc1, idc2 = i*2, (i+1) % N * 2
        x1, y1 = vertices[idc1], vertices[idc1+1]
        x2, y2 = vertices[idc2], vertices[idc2+1]
        # x2, y2 = vertices[(i + 1) % N * 2], vertices[(i + 1) % N + 1]
        dx = x2 - x1
        dy = y2 - y1
        Ai = cs.vertcat(dy, -dx)
        bi = x1 * dy - y1 * dx
        # print(i, [x1,y1],[x2, y2], bi-cs.dot(Ai, p))
        hp_vals += [bi-cs.dot(Ai, p)]
    return cs.vertcat(*hp_vals)

class Mpc:

    def __init__(self, params, robot):
        self.build_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mpc_build')
        # Load parameters
        self.build_params = None
        self.robot = robot
        self.set_build_params(params)
        rebuild, self.build_name = self.get_build_version()
        self.generate_sol_fun()
        # Build if different from latest build
        if rebuild:
            self.build()
        else:
            print("Found MPC build: {}".format(self.build_name))

        sys.path.insert(1, os.path.join(self.build_dir, self.build_name))
        optimizer = __import__(self.build_name)
        self.solver = optimizer.solver()
        self.reset()

    def reset(self):
        self.sol_prev = None

    def set_build_params(self, params):
        self.build_params = {
            'mode': params['build_mode'],
            'N': params['N'],
            'dt': params['dt'],
            'solver_tol': params['solver_tol'],
            'solver_max_time': params['solver_max_time'],
            'solver_max_inner_iterations': params['solver_max_inner_iterations'],
            'solver_max_outer_iterations': params['solver_max_outer_iterations'],
            'x_min': self.robot.x_min,
            'x_max': self.robot.x_max,
            'u_min': self.robot.u_min,
            'u_max': self.robot.u_max,
            'nx': self.robot.nx,
            'nu': self.robot.nu,
            'np': 2,
            'robot_model': self.robot.__class__.__name__,
            'n_pol': params['n_pol'],
            'integration_method': params['integration_method'],
            'w_max': self.robot.vmax,
            'max_No_ell': params['max_No_ell'],
            'max_No_pol': params['max_No_pol'],
            'max_No_vert': params['max_No_vert'],
            'N_obs_predict': params['N_obs_predict'],
            'dlp_con': params['dlp_con'],
            'xaeN_con': params['xaeN_con'],
        }

    def get_build_version(self):
        builds = [name for name in os.listdir(self.build_dir)
                  if os.path.isdir(os.path.join(self.build_dir, name))]
        for build_ver in builds:
            if not os.path.isfile(os.path.join(self.build_dir, build_ver, 'build_params.yaml')):
                continue
            with open(os.path.join(self.build_dir, build_ver, 'build_params.yaml'), 'r') as file:
                build_ver_params = yaml.load(file, Loader=yaml.FullLoader)
            if self.build_params == build_ver_params:
                return False, build_ver
        ver = 0
        while 'ver' + str(ver) in os.listdir(self.build_dir):
            ver += 1
        return True, 'ver' + str(ver)

    def cs_obstacle_evaluation(self, obs_par, x, k):
        n_ell_par = 3 * self.build_params['max_No_ell'] * (1 + self.build_params['N_obs_predict'])
        obs_val = []
        for i in range(self.build_params['max_No_ell']):
            j = i * 3 * (1 + self.build_params['N_obs_predict'])
            time_idx = min(k, self.build_params['N_obs_predict']-1)
            include_obs = obs_par[j]
            ell_axs = obs_par[j + 1:j + 3]
            ell_pos = obs_par[j + 3 + 3 * time_idx:j + 5 + 3 * time_idx]
            ell_rot = obs_par[j + 5 + 3 * time_idx]
            ell_val = ellipse_function(self.robot.h(x), ell_pos, ell_rot, ell_axs)
            obs_val += [cs.fmax(include_obs * ell_val, 0)]

        for i in range(self.build_params['max_No_pol']):
            j = n_ell_par + i * self.build_params['max_No_vert'] * 2
            hp_vals = convex_polygon_function(self.robot.h(x), obs_par[j : j + 2 * self.build_params['max_No_vert']], self.build_params['max_No_vert'])
            obs_val += [cs.fmax(cs.mmin(hp_vals),0)]
        return cs.vertcat(*obs_val)

    def is_feasible(self, sol, x0, path_pol, obs_par, params, d=False):
        con = self.sol2con(sol, x0, path_pol, params)
        sol_min, sol_max = self.sol_bounds()
        con_min, con_max = self.con_bounds()

        obs_val = self.sol2obsval(sol, x0, obs_par)

        eps = 1.e-4

        def in_range(val, min, max):
            return ((np.array(val) >= np.array(min)) & (np.array(val) <= np.array(max))).all()

        sol_ok = True

        if not in_range(sol, sol_min - eps, sol_max + eps):
            sol_ok = False
        if not in_range(con, con_min - eps, con_max + eps):
            sol_ok = False
        if not in_range(obs_val, 0 - eps, 0 + eps):
            sol_ok = False

        return sol_ok

        # if not in_range(u, u_min, u_max):
        #     sol_ok = False
        #     if d:
        #         print("[MPC]: Bad u.")
        #         print(u)
        # if not in_range(w, w_min, w_max):
        #     sol_ok = False
        #     if d:
        #         print("[MPC]: Bad w. Not in [{:.4f}, {:.4f}]".format(0, self.build_params['dp_max']))
        #         print(w)
        # if not in_range(x, x_min, x_max):
        #     sol_ok = False
        #     if d:
        #         print("[MPC]: Bad x")
        #         print(x)
        # if not in_range(s, s_min, s_max):
        #     sol_ok = False
        #     if d:
        #         print("[MPC]: Bad sN {:.4f} > {:.4f}".format(s[-1], s_max[0]))
        # return sol_ok

    def error(self, x, s, path_pol):
        p_ref = cs.vertcat(*pol2pos(path_pol, s, self.build_params['n_pol']))
        return cs.norm_2(p_ref - self.robot.h(x))

    def base_solution(self, x0, path_pol, w0):
        u = [0] * (self.build_params['N'] * self.build_params['nu'])
        xa0 = x0.copy()
        w = [0] * (self.build_params['N'])
        w[0] = w0

        if self.build_params['robot_model'] == 'Unicycle':
            p_ref = pol2pos(path_pol, w0*self.build_params['dt'], self.build_params['n_pol'])
            k1, k2 = 10, 2
            # k1, k2 = 0.15, 0.3

            x = x0.copy()
            for i in range(self.build_params['N']):
                e = np.array(x[:2]) - np.array(p_ref)
                th = x[2]

                r_th = np.arctan2(e[1], e[0]) + np.pi
                # E Theta in (-pi, pi]
                e_th = r_th - th
                e_th0 = e_th
                while e_th > np.pi:
                    e_th -= 2 * np.pi
                while e_th <= -np.pi:
                    e_th += 2 * np.pi

                ui = [
                    -k1 * (e[0] * np.cos(th) + e[1] * np.sin(th)),
                    k2 * (e_th)
                ]

                if self.robot.u_min[0] == 0 and ui[0] < 0:
                    ui[0] = 0
                    u0_sig = 1
                else:
                    u0_sig = ui[0] / self.robot.u_max[0] if ui[0] >= 0 else ui[0] / self.robot.u_min[0]
                u1_sig = ui[1] / self.robot.u_max[1] if ui[1] >= 0 else ui[1] / self.robot.u_min[1]

                sig = max(u0_sig, u1_sig, 1)
                if sig > 1:
                    if u0_sig > u1_sig:
                        ui[0] = self.robot.u_max[0] if ui[0] > 0 else self.robot.u_min[0]
                        ui[1] /= sig
                    else:
                        ui[0] /= sig
                        ui[1] = self.robot.u_max[1] if ui[1] > 0 else self.robot.u_min[1]
                u[i * self.build_params['nu']:(i+1) * self.build_params['nu']] = ui
                # print(e, e_th, ui)
                x, _ = self.robot.move(x, ui, self.build_params['dt'])

        ua = u

        return xa0 + u + ua + w

    def discrete_integration(self, f, x, u):
        dt = self.build_params['dt']
        if self.build_params['integration_method'] == 'euler':
            return x + f(x, u) * dt
        if self.build_params['integration_method'] == 'RK4':
            k1 = f(x, u)
            k2 = f(x + dt / 2 * k1, u)
            k3 = f(x + dt / 2 * k2, u)
            k4 = f(x + dt * k3, u)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        raise NotImplemented

    def generate_sol_fun(self):
        # Initialize
        sol_sym = cs.SX.sym('sol', self.build_params['nx'] + (2 * self.build_params['nu'] + 1) * self.build_params['N'])
        x0_sym = cs.SX.sym('x0', self.build_params['nx'])
        path_pol_sym = cs.SX.sym('path_pol', self.build_params['np'] * (self.build_params['n_pol'] + 1))

        n_ell_par = 3 * self.build_params['max_No_ell'] * (1 + self.build_params['N_obs_predict'])
        n_pol_par = self.build_params['max_No_pol'] * self.build_params['max_No_vert'] * 2
        obs_par_sym = cs.SX.sym('obs_par', n_ell_par + n_pol_par)
        # obs_par_sym = cs.SX.sym('obs_par', self.build_params['max_No_ell'] * self.build_params['N_obs_predict'] * 6 + self.build_params['max_No_pol'] * self.build_params['max_No_vert'] * 2)
        cost_params = {'Q': self.build_params['nx'], 'R': self.build_params['nu'], 'K': 1,
                       'S': self.build_params['nu'], 'T': 1, 'mu': 1}
        cost_params_sym = cs.SX.sym("cost_params", sum(list(cost_params.values())))
        # Exchange parameter dimension with SX variable
        p_idx = 0
        for key, dim in cost_params.items():
            cost_params[key] = cost_params_sym[p_idx:p_idx + dim]
            p_idx += dim
        # Initialize
        xa0, u, ua, w = self.sol2xa0uuaw(sol_sym)
        u_k, ua_k, w_k = u[:self.build_params['nu']], ua[:self.build_params['nu']], w[0]
        x_k, xa_k, s_k = x0_sym, xa0, 0
        ea_k = self.error(xa_k, s_k, path_pol_sym)
        obs_val_k = self.cs_obstacle_evaluation(obs_par_sym, x_k, 0)
        obs_val_a_k = self.cs_obstacle_evaluation(obs_par_sym, xa_k, 0)
        # lp_k = cost_params['K'] * ea_k ** 2 + cs.dot(cost_params['S'], ua_k * ua_k) + cost_params['T'] * (w_k - self.build_params['dp_max']) ** 2
        lp_k = cost_params['K'] * ea_k ** 2 + cost_params['T'] * (w_k - self.build_params['w_max']) ** 2
        x, s, ea, xa, obs_val, obs_val_a = x_k, s_k, ea_k, xa_k, obs_val_k, obs_val_a_k
        dlp = []
        # Loop over time steps
        cs_f = lambda x,u: cs.vertcat(*self.robot.f(x,u))
        cs_fs = lambda s,w: w
        for k in range(self.build_params['N']):
            # Current control variables
            u_k = u[k * self.build_params['nu']:(k + 1) * self.build_params['nu']]
            ua_k = ua[k * self.build_params['nu']:(k + 1) * self.build_params['nu']]
            w_k = w[k]
            # Integrate one step
            x_k = self.discrete_integration(cs_f, x_k, u_k)
            s_k = self.discrete_integration(cs_fs, s_k, w_k)
            xa_k = self.discrete_integration(cs_f, xa_k, ua_k)
            lp_prev_k = lp_k
            # Extract error and obstacle values
            ea_k = self.error(xa_k, s_k, path_pol_sym)
            obs_val_k = self.cs_obstacle_evaluation(obs_par_sym, x_k, k+1)
            obs_val_a_k = self.cs_obstacle_evaluation(obs_par_sym, xa_k, k+1)

            # lp_k = cost_params['K'] * ea_k ** 2 + cs.dot(cost_params['S'], ua_k * ua_k) + cost_params['T'] * (w_k - self.build_params['dp_max']) ** 2
            lp_k = cost_params['K'] * ea_k ** 2 + cost_params['T'] * (w_k - self.build_params['w_max']) ** 2
            dlp_k = lp_k - lp_prev_k

            # Store current state
            x = cs.vertcat(x, x_k)
            xa = cs.vertcat(xa, xa_k)
            s = cs.vertcat(s, s_k)
            ea = cs.vertcat(ea, ea_k)
            obs_val = cs.vertcat(obs_val, obs_val_k)
            obs_val_a = cs.vertcat(obs_val_a, obs_val_a_k)
            dlp = cs.vertcat(dlp, dlp_k)

        # Define constraint vector
        con = cs.vertcat(s, x)
        if self.build_params['xaeN_con']:
            con = cs.vertcat(con, x_k-xa_k)
        if self.build_params['dlp_con']:
            con = cs.vertcat(con, dlp)
        # con = cs.vertcat(s, x, x_k-xa_k, dlp)
        # con = cs.vertcat(s, x, x_k-xa_k)

        u_target = cs.SX([self.build_params['w_max'], 0])
        u_err = cs.repmat(u_target,self.build_params['N']) - ua
        xae = x - xa
        uae = u - ua
        we = w - self.build_params['w_max']
        # Define costs
        Q = cs.repmat(cost_params['Q'], self.build_params['N'] + 1)
        R = cs.repmat(cost_params['R'], self.build_params['N'])
        K = cs.repmat(cost_params['K'], self.build_params['N'] + 1)
        S = cs.repmat(cost_params['S'], self.build_params['N'])
        T = cs.repmat(cost_params['T'], self.build_params['N'])
        # Note s_cost is normalized for easier tuning
        la_cost = cs.dot(Q, xae * xae) + cs.dot(R, uae * uae)
        lo_cost = cs.dot(K, ea * ea) + cs.dot(S, u_err * u_err)
        lw_cost = cs.dot(T, we ** 2)
        obs_cost = 0.5 * cost_params['mu'] * (cs.sum1(obs_val * obs_val) + cs.sum1(obs_val_a * obs_val_a))
        # Define constraints
        self.cs_sol2state = cs.Function('cs_sol2state', [sol_sym, x0_sym, path_pol_sym], [x, s, xa], ['sol', 'x0', 'path_pol'], ['x', 's', 'xa'])
        self.cs_sol2cost = cs.Function('cs_sol2cost', [sol_sym, x0_sym, path_pol_sym, obs_par_sym, cost_params_sym], [la_cost, lo_cost, lw_cost, obs_cost], ['sol', 'x0', 'path_pol', 'obs_par', 'cost_params'], ['la_cost', 'lo_cost', 'lw_cost', 'obs_cost'])
        self.cs_sol2con = cs.Function('cs_sol2con', [sol_sym, x0_sym, path_pol_sym, cost_params_sym], [con], ['sol', 'x0', 'path_pol', 'cost_params'], ['con'])
        self.cs_sol2obsval = cs.Function('cs_sol2obsval', [sol_sym, x0_sym, obs_par_sym], [obs_val], ['sol', 'x0', 'obs_par'], ['obs_val'])

    def sol_bounds(self):
        xa0_min = np.tile(-np.inf, self.build_params['nx'])
        u_min = np.tile(self.build_params['u_min'], self.build_params['N'])
        ua_min = np.tile(-np.inf, self.build_params['nu'] * self.build_params['N'])
        w_min = np.zeros(self.build_params['N'])
        xa0_max = np.tile(np.inf, self.build_params['nx'])
        u_max = np.tile(self.build_params['u_max'], self.build_params['N'])
        ua_max = np.tile(np.inf, self.build_params['nu'] * self.build_params['N'])
        w_max = np.tile(self.build_params['w_max'], self.build_params['N'])
        return np.concatenate((xa0_min, u_min, ua_min, w_min)), np.concatenate((xa0_max, u_max, ua_max, w_max))

    def con_bounds(self):
        s_min = np.zeros(self.build_params['N'] + 1)
        x_min = np.tile(self.build_params['x_min'], self.build_params['N'] + 1)
        xaeN_min = np.tile(-0.1, self.build_params['nx'])
        dlp_min = np.tile(-np.inf, self.build_params['N'])
        s_max = np.tile(self.build_params['w_max'] * self.build_params['dt'] * self.build_params['N'], self.build_params['N'] + 1)
        x_max = np.tile(self.build_params['x_max'], self.build_params['N'] + 1)
        xaeN_max = np.tile(0.1, self.build_params['nx'])
        dlp_max = np.tile(0, self.build_params['N'])
        con_min, con_max = np.concatenate((s_min, x_min)), np.concatenate((s_max, x_max))
        if self.build_params['xaeN_con']:
            con_min, con_max = np.concatenate((con_min, xaeN_min)), np.concatenate((con_max, xaeN_max))
        if self.build_params['dlp_con']:
            con_min, con_max = np.concatenate((con_min, dlp_min)), np.concatenate((con_max, dlp_max))
        return con_min, con_max

    def sol2state(self, sol, x0, path_pol):
        x, s, xa = self.cs_sol2state(sol, x0, path_pol)
        return np.array(x).flatten(), np.array(s).flatten(), np.array(xa).flatten()

    def sol2con(self, sol, x0, path_pol, params):
        cost_params = params['Q'] + params['R'] + [params['K']] + params['S'] + [params['T'], params['mu']]
        return np.array(self.cs_sol2con(sol, x0, path_pol, cost_params)).flatten()

    def sol2obsval(self, sol, x0, obs_par):
        return np.array(self.cs_sol2obsval(sol, x0, obs_par)).flatten()

    def sol2cost(self, sol, x0, path_pol, obs_par, params):
        cost_params = params['Q'] + params['R'] + [params['K']] + params['S'] + [params['T'], params['mu']]
        la_cost, lo_cost, lw_cost, obs_cost = self.cs_sol2cost(sol, x0, path_pol, obs_par, cost_params)
        return {'la_cost': float(la_cost), 'lo_cost': float(lo_cost), 'lw_cost': float(lw_cost), 'obs_cost': float(obs_cost)}

    def sol2xa0uuaw(self, sol):
        xa0_len, u_len, ua_len, w_len = self.build_params['nx'], self.build_params['nu'] * self.build_params['N'], \
                                        self.build_params['nu'] * self.build_params['N'], self.build_params['N']
        u_idx = xa0_len
        ua_idx = u_idx + u_len
        w_idx = ua_idx + ua_len
        xa0 = sol[:u_idx]
        u = sol[u_idx:ua_idx]
        ua = sol[ua_idx:w_idx]
        w = sol[w_idx:]
        return xa0, u, ua, w

    def build(self):
        # Build parametric optimizer
        # ------------------------------------

        n_ell_par = 3 * self.build_params['max_No_ell'] * (1 + self.build_params['N_obs_predict'])
        n_pol_par = self.build_params['max_No_pol'] * self.build_params['max_No_vert'] * 2

        params = {'x0': self.build_params['nx'], 'path_pol': self.build_params['np'] * (self.build_params['n_pol'] + 1),
                  'Q': self.build_params['nx'], 'R': self.build_params['nu'], 'K': 1,
                   'S': self.build_params['nu'], 'T': 1, 'mu': 1,
                  'obs_par': n_ell_par + n_pol_par}
        par_dim = sum(list(params.values()))
        # Exchange parameter dimension with value
        par = cs.SX.sym("par", par_dim)  # Parameters
        p_idx = 0
        for key, dim in params.items():
            params[key] = par[p_idx:p_idx + dim]
            p_idx += dim

        # Define solution vector
        sol = cs.SX.sym('sol', self.build_params['nx'] + (2 * self.build_params['nu'] + 1) * self.build_params['N'])

        # Cost
        cost_params = cs.vertcat(params['Q'], params['R'], params['K'], params['S'], params['T'], params['mu'])
        la_cost, lo_cost, lw_cost, obs_cost = self.cs_sol2cost(sol, params['x0'], params['path_pol'], params['obs_par'], cost_params)
        cost = la_cost + lo_cost + lw_cost + obs_cost

        # Constraints
        sol_min, sol_max = self.sol_bounds()
        con_min, con_max = self.con_bounds()
        sol_bounds = og.constraints.Rectangle(sol_min.tolist(), sol_max.tolist())
        agl_con = self.cs_sol2con(sol, params['x0'], params['path_pol'], cost_params)
        agl_con_bounds = og.constraints.Rectangle(con_min.tolist(), con_max.tolist())
        # Setup builder
        problem = og.builder.Problem(sol, par, cost) \
            .with_constraints(sol_bounds) \
            .with_aug_lagrangian_constraints(agl_con, agl_con_bounds)
        build_config = og.config.BuildConfiguration() \
            .with_build_directory(self.build_dir) \
            .with_build_mode(self.build_params['mode']) \
            .with_build_python_bindings()
        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(self.build_name)
        solver_config = og.config.SolverConfiguration() \
            .with_tolerance(self.build_params['solver_tol']) \
            .with_max_duration_micros(self.build_params['solver_max_time'] * 1000) \
            .with_max_inner_iterations(self.build_params['solver_max_inner_iterations']) \
            .with_max_outer_iterations(self.build_params['solver_max_outer_iterations'])
        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                  meta,
                                                  build_config,
                                                  solver_config) \
            .with_verbosity_level(1)
        builder.build()
        print('')

        # Save build params
        with open(os.path.join(self.build_dir, self.build_name, 'build_params.yaml'), 'w') as file:
            yaml.dump(self.build_params, file)


    def run(self, x0, path_pol, params, obs_par, init_guess):
        p = x0 + path_pol + params['Q'] + params['R'] + [params['K']] + params['S'] + [params['T'], params['mu']] + obs_par
        if init_guess is None:# or not self.is_feasible(self.sol_prev, x0, path_pol, 2*e_max, lam_rho):
            # Use base solution as initial guess
            init_guess = self.base_solution(x0, path_pol, 0.1)

        # Run solver
        solution_data = self.solver.run(p=p, initial_guess=init_guess)

        return solution_data

        base_sol = [0] * (self.build_params['nx'] + (2 * self.build_params['nu'] + 1) * self.build_params['N'])
        base_sol[:self.build_params['nx']] = x0
        base_sol[-self.build_params['N']] = self.build_params['dp_max']


        if self.sol_prev is None:# or not self.is_feasible(self.sol_prev, x0, path_pol, params):
            # Use base solution as initial guess
            self.sol_prev = base_sol

        # Run solver
        solution_data = self.solver.run(p=p, initial_guess=self.sol_prev)

        if solution_data is None:
            sol, exit_status = base_sol, "TrivialSolution"
            self.sol_prev = None
        else:
            sol, exit_status = solution_data.solution, solution_data.exit_status
            # self.sol_prev = sol
            x, s, xa = self.sol2state(sol, x0, path_pol)

            xa0, u, ua, w = self.sol2xa0uuaw(sol)

            self.sol_prev = xa[self.build_params['nx']:2*self.build_params['nx']].tolist() + \
                            u[self.build_params['nu']:] + [0] * self.build_params['nu'] + \
                            ua[self.build_params['nu']:] + [0] * self.build_params['nu'] + \
                            w[1:] + [0]

        sol_feasible = self.is_feasible(sol, x0, path_pol, params, d=verbosity>0)

        return sol, sol_feasible, exit_status