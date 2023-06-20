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
            'w_max': self.robot.vmax
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

    def is_feasible(self, sol, x0, path_pol, e_max, lam_rho, d=False):
        u, w = self.sol2uds(sol)
        x, s, e = self.sol2state(sol, x0, path_pol)
        u_min, w_min, s1_lamrho_min, sN_min, x_min, e_max_min = self.min_bounds()
        u_max, w_max, s1_lamrho_max, sN_max,  x_max, e_max_max = self.max_bounds()

        eps = 1e-3

        sol_ok = True
        if not all((u_min <= u) & (u <= u_max)):
            sol_ok = False
            if d:
                print("[MPC]: Bad u")
                print(u)
        if not all((w_min <= w) & (w <= w_max)):
            sol_ok = False
            if d:
                print("[MPC]: Bad w")
                print(w)
        if not all((x_min <= x) & (x <= x_max)):
            sol_ok = False
            if d:
                print("[MPC]: Bad x")
                print(x)
        if s1_lamrho_min - eps > s[1] - lam_rho:
            sol_ok = False
            if d:
                print("[MPC]: Bad s1 {:.4f} < {:.4f}".format(s[1], lam_rho))
        if not ((sN_min <= s[-1]) and (s[-1] <= sN_max + 0.1)):
            sol_ok = False
            if d:
                print("[MPC]: Bad sN {:.4f} > {:.4f}".format(s[-1], sN_max))
        if not all((e_max_min <= e - e_max) & (e - e_max <= e_max_max)):
            sol_ok = False
            if d:
                print("[MPC]: Bad e (e_max: {:.4f})".format(e_max))
                print(e)
        return sol_ok

    def error(self, x, s, path_pol):
        p_ref = cs.vertcat(*pol2pos(path_pol, s, self.build_params['n_pol']))
        return cs.norm_2(p_ref - self.robot.h(x))

    def base_solution(self, x0, path_pol, lam_rho):
        u = [0] * (self.build_params['N'] * self.build_params['nu'])
        w = [0] * (self.build_params['N'])
        w[0] = lam_rho / self.build_params['dt']

        if self.build_params['robot_model'] == 'Unicycle':
            p_ref = pol2pos(path_pol, lam_rho, self.build_params['n_pol'])
            k1, k2 = 2, 2
            # k1, k2 = 1.25, 0.3

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

                x, _ = self.robot.move(x, ui, self.build_params['dt'])

        return u + w

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
        sol_sym = cs.SX.sym('sol', (self.build_params['nu'] + 1) * self.build_params['N'])
        x0_sym = cs.SX.sym('x0', self.build_params['nx'])
        path_pol_sym = cs.SX.sym('path_pol', self.build_params['np'] * (self.build_params['n_pol'] + 1))
        cost_params = {'cs': 1, 'ce': 1, 'R': self.build_params['nu'], 'DR': self.build_params['nu'],
                       'u_prev': self.build_params['nu']}
        cost_params_sym = cs.SX.sym("cost_params", sum(list(cost_params.values())))
        # Exchange parameter dimension with SX variable
        p_idx = 0
        for key, dim in cost_params.items():
            cost_params[key] = cost_params_sym[p_idx:p_idx + dim]
            p_idx += dim
        # Initialize
        u, w = self.sol2uds(sol_sym)
        x_k, s_k = x0_sym, 0
        e_k = self.error(x_k, s_k, path_pol_sym)
        x, s, e = x_k, s_k, e_k
        # Loop over time steps
        for k in range(self.build_params['N']):
            # Current control variables
            u_k = u[k * self.build_params['nu']:(k + 1) * self.build_params['nu']]
            # Integrate one step
            x_k = self.discrete_integration(lambda x, u: cs.vertcat(*self.robot.f(x,u)), x_k, u_k)
            s_k = self.discrete_integration(lambda s, w: w, s_k, w[k])
            e_k = self.error(x_k, s_k, path_pol_sym)
            # Store current state
            x = cs.vertcat(x, x_k)
            s = cs.vertcat(s, s_k)
            e = cs.vertcat(e, e_k)
        # Define costs
        u_target = cs.SX([self.build_params['w_max'], 0])
        u_err = cs.repmat(u_target,self.build_params['N']) - u
        du = u - cs.vertcat(cost_params['u_prev'], u[:-self.build_params['nu']])
        R = cs.repmat(cost_params['R'], self.build_params['N'])
        DR = cs.repmat(cost_params['DR'], self.build_params['N'])
        w_cost = cost_params['cs'] * cs.sum1(self.build_params['w_max'] - w)
        e_cost = cost_params['ce'] * cs.sum1(e)
        # u_cost = cs.dot(R, u * u)
        u_cost = cs.dot(R, u_err * u_err)
        du_cost = cs.dot(DR, du * du)
        # Define constraints
        self.cs_sol2state = cs.Function('cs_sol2state', [sol_sym, x0_sym, path_pol_sym], [x, s, e], ['sol', 'x0', 'path_pol'], ['x', 's', 'e'])
        self.cs_sol2cost = cs.Function('cs_sol2cost', [sol_sym, x0_sym, path_pol_sym, cost_params_sym], [w_cost, e_cost, u_cost, du_cost], ['sol', 'x0', 'path_pol', 'cost_params'], ['s_cost', 'e_cost', 'u_cost', 'du_cost'])

    def min_bounds(self):
        u_min = np.tile(self.build_params['u_min'], self.build_params['N'])
        w_min = np.zeros(self.build_params['N'])
        s1_lamrho_min = 0
        sN_min = 0
        x_min = np.tile(self.build_params['x_min'], self.build_params['N'] + 1)
        e_max_min = np.tile(-np.inf, self.build_params['N'] + 1)
        return u_min, w_min, s1_lamrho_min, sN_min, x_min, e_max_min

    def max_bounds(self):
        u_max = np.tile(self.build_params['u_max'], self.build_params['N'])
        # ds_max = np.tile(np.inf, self.build_params['N'])
        w_max = np.tile(self.build_params['w_max'], self.build_params['N'])
        s1_lamrho_max = np.inf
        sN_max = self.build_params['w_max'] * self.build_params['N'] * self.build_params['dt']
        x_max = np.tile(self.build_params['x_max'], self.build_params['N'] + 1)
        e_max_max = np.zeros(self.build_params['N'] + 1)
        return u_max, w_max, s1_lamrho_max, sN_max, x_max, e_max_max

    def sol2state(self, sol, x0, path_pol):
        x, s, e = self.cs_sol2state(sol, x0, path_pol)
        # cs.MX to [float]
        return np.array(x).flatten(), np.array(s).flatten(), np.array(e).flatten()

    def sol2cost(self, sol, x0, path_pol, params, u_prev):
        cost_params = [params['cs'], params['ce'], params['R'][0], 0, params['DR'][0], 0] + u_prev
        cost_params[2:6] = [params['R'][0], 0, params['DR'][0], 0]
        s_cost, e_cost, u1_cost, ud1_cost = self.cs_sol2cost(sol, x0, path_pol, cost_params)
        cost_params[2:6] = [0, params['R'][1], 0, params['DR'][1]]
        _, _, u2_cost, ud2_cost = self.cs_sol2cost(sol, x0, path_pol, cost_params)
        return {'s': float(s_cost), 'e': float(e_cost), 'u': float(u1_cost + u2_cost), 'ud': float(ud1_cost + ud2_cost),
                'u1': float(u1_cost), 'ud1': float(ud1_cost), 'u2': float(u2_cost), 'ud2': float(ud2_cost)}

    def sol2uds(self, sol):
        u = sol[:self.build_params['nu'] * self.build_params['N']]
        ds = sol[self.build_params['nu'] * self.build_params['N']:]
        return u, ds

    def build(self):
        # Build parametric optimizer
        # ------------------------------------
        params = {'x0': self.build_params['nx'], 'u_prev': self.build_params['nu'],
                  'path_pol': self.build_params['np'] * (self.build_params['n_pol'] + 1),
                  'cs': 1, 'ce': 1, 'R': self.build_params['nu'], 'DR': self.build_params['nu'], 'e_max': 1,
                  'lam_rho': 1}
        par_dim = sum(list(params.values()))

        # Exchange parameter dimension with value
        par = cs.SX.sym("par", par_dim)  # Parameters
        p_idx = 0
        for key, dim in params.items():
            params[key] = par[p_idx:p_idx + dim]
            p_idx += dim

        # Initialize
        sol = cs.SX.sym('sol', (self.build_params['nu'] + 1) * self.build_params['N'])

        # Cost
        cost_params = cs.vertcat(params['cs'], params['ce'], params['R'], params['DR'], params['u_prev'])
        s_cost, e_cost, u_cost, du_cost = self.cs_sol2cost(sol, params['x0'], params['path_pol'], cost_params)
        cost = s_cost + e_cost + u_cost + du_cost

        # Constraints
        x, s, e = self.cs_sol2state(sol, params['x0'], params['path_pol'])
        u_min, w_min, s1_lamrho_min, sN_min, x_min, e_max_min = self.min_bounds()
        u_max, w_max, s1_lamrho_max, sN_max, x_max, e_max_max = self.max_bounds()
        sol_bounds = og.constraints.Rectangle(u_min.tolist() + w_min.tolist(), u_max.tolist() + w_max.tolist())
        agl_con = cs.vertcat(s[1] - params['lam_rho'],
                             s,
                             x,
                             e - params['e_max'])
        agl_con_bounds = og.constraints.Rectangle([s1_lamrho_min] + [sN_min] * (self.build_params['N'] + 1) +
                                                  x_min.tolist() + e_max_min.tolist(),
                                                  [s1_lamrho_max] + [sN_max] * (self.build_params['N'] + 1) +
                                                  x_max.tolist() + e_max_max.tolist())

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

    def shift_sol(self, sol):
        return sol[self.build_params['nu']:self.build_params['nu'] * self.build_params['N']] + \
               sol[self.build_params['nu'] * (self.build_params['N'] - 1) : self.build_params['nu'] * self.build_params['N']] \
               + sol[self.build_params['nu'] * self.build_params['N']:]

    def run(self, x0, u_prev, path_pol, params, e_max, lam_rho, init_guess=None, verbosity=0):

        p = x0 + u_prev + path_pol + [params['cs'], params['ce']] + params['R'] + params['DR'] + \
            [0.9*e_max, 1.1*lam_rho]


        if init_guess is None:# or not self.is_feasible(self.sol_prev, x0, path_pol, 2*e_max, lam_rho):
            # Use base solution as initial guess
            init_guess = self.base_solution(x0, path_pol, lam_rho)
        # Run solver
        return self.solver.run(p=p, initial_guess=init_guess)
        solution_data = self.solver.run(p=p, initial_guess=init_guess)

        if solution_data is None or not self.is_feasible(solution_data.solution, x0, path_pol, e_max, lam_rho, d=verbosity>0):
            sol, exit_status = self.base_solution(x0, path_pol, lam_rho), "BaseSolution"
        else:
            sol = solution_data.solution
            exit_status = solution_data.exit_status

        # self.sol_prev = sol[self.build_params['nu']:self.build_params['nu'] * self.build_params['N']] + \
        #                 sol[self.build_params['nu'] * (self.build_params['N'] - 1):self.build_params['nu'] *
        #                                                                            self.build_params['N']] \
        #                 + sol[self.build_params['nu'] * self.build_params['N']:]
        sol_feasible = self.is_feasible(sol, x0, path_pol, e_max, lam_rho, d=verbosity > 0)

        return sol, sol_feasible, exit_status
