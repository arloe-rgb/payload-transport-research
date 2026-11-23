import yaml
import torch
import math
from torch.func import grad, jacfwd

class allocator:

    def __init__(
        self, config, B_mat_func, control_cmd_func, get_traj_func,
        predictor=None, adaptive_accel_cmd_func=None,
    ):

        # Import config parameters
        allocator = config["allocation"]
        sim = config["SIM"]

        # Import refs to other class functions
        self.B_mat = B_mat_func
        self.control_cmd = control_cmd_func
        self.get_trajectory = get_traj_func
        self.predictor = predictor
        self.adaptive_accel_cmd = adaptive_accel_cmd_func

        # Number of UAVs
        self.N = sim["N"]

        # # Maximum force for each UAV
        # self.F_max = self.u.new_tensor(allocator["F_max"])

        # Init decison variable u
        self.u = torch.zeros(self.N*2, dtype=torch.float64, device="cpu")
        self.F_max = self.u.new_tensor(allocator["F_max"])
        self.u[1::2] = 0.5 * self.F_max
        self.min_cable_angle_sin = math.sin(math.radians(allocator["min_cable_angle_deg"]))

        # multipliers for equality constraints
        self.lamda = self.u.new_zeros(3)

        # log barrier scalar
        self.log_scalar = allocator["log_scalar"]

        # Optimization weights (one per UAV)
        self.weights = self.u.new_tensor(allocator["weights"])

        self.Gamma = torch.diag(self.u.new_tensor(allocator["gamma"]))

    def objective_function(self, u_control, theta):

        # reshape u for easier computation
        u = u_control.reshape(self.N, 2)

        # Calc (||u_i||_2)^2 first
        force_mag = torch.sum(u**2, dim=1)

        # 1/2 s^T * ||u||^2
        quad_cost = 0.5 * torch.dot(self.weights, force_mag)

        # Inequality constraints 
        force_mag_constr = force_mag - self.F_max**2   # Max force constraints
        force_dir_constr = -u[:, 1]   # Force direction constraints

        # Cable elevation margin relative to the payload surface
        normal = torch.stack((-torch.sin(theta), torch.cos(theta)))
        angle_margin = u @ normal - torch.sqrt(force_mag) * self.min_cable_angle_sin

        # Log barrier cost function
        barrier_cost = -self.log_scalar*(
            torch.sum(torch.log(-force_mag_constr))
            + torch.sum(torch.log(-force_dir_constr))
            + torch.log(angle_margin).sum()
        )

        return quad_cost + barrier_cost

    def equality_constraints(self, B, u, tau_d):

        equal_constr = B @ u - tau_d

        return equal_constr

    def lagrangian(self, t, u, lamda, x):

        B = self.B_mat(x[2])

        x_ref, qddot_ref = self.get_trajectory(t)

        # calls control_cmd here because to fix issue of u_ff needing explicit expressions for B_dot and tau_d_dot (not my favorite as it does not keep the architecture modular)
        tau_d = self.control_cmd(x, x_ref, qddot_ref)

        obj_func = self.objective_function(u, x[2])

        eq_constr = self.equality_constraints(B, u, tau_d)

        L = obj_func + torch.dot(lamda, eq_constr)

        return L

    def FOO_vector(self, t, u, lamda, x):

        dL_du, dL_dlamda = grad(self.lagrangian, argnums=(1, 2))(t, u, lamda, x)

        return torch.cat((dL_du, dL_dlamda))

    def hessian(self, t, u, lamda, x):

        dFOO_du, dFOO_dlamda = jacfwd(self.FOO_vector, argnums=(1, 2))(t, u, lamda, x)

        return torch.cat((dFOO_du, dFOO_dlamda), dim=1)

    def u_ff(self, t, u, lamda, x, x_dot):

        # time derivative of FOO_vector
        dFOO_dt = jacfwd(self.FOO_vector, argnums=0)(t, u, lamda, x)

        # state derivative of FOO_vector
        dFOO_dx = jacfwd(self.FOO_vector, argnums=3)(t, u, lamda, x)

        uff = dFOO_dt + dFOO_dx @ x_dot

        return uff

    def allocator_dynamics(self, t, u, lamda, x, x_dot):

        FOO_vector = self.FOO_vector(t, u, lamda, x)

        hessian = self.hessian(t, u, lamda, x)

        uff = self.u_ff(t, u, lamda, x, x_dot)

        ff_term = self.Gamma @ FOO_vector + uff

        # constructing the dynamics of the decision variables and multipliers
        ud_ld = torch.linalg.solve(hessian, -ff_term)

        return ud_ld

    def allocate(self, t, x, x_dot, dt):

        u = self.u
        lamda = self.lamda

        k1 = self.allocator_dynamics(t, u, lamda, x, x_dot)
        u_mid = u + (dt/2)*k1[:2*self.N]
        lamda_mid = lamda + (dt/2)*k1[2*self.N:]
        k2 = self.allocator_dynamics(t + dt/2, u_mid, lamda_mid, x, x_dot)

        self.u = u + dt*k2[:2*self.N]
        self.lamda = lamda + dt*k2[2*self.N:]

        return self.u

    def adaptive_lagrangian(self, t, u, lamda, x, x_hat, w_hat):
        x_ref, qddot_ref = self.get_trajectory(t)

        # Commanded and estimated payload acceleration
        qddot_cmd = self.adaptive_accel_cmd(x, x_hat, x_ref, qddot_ref)
        phi = self.predictor.regressor(x, u)
        qddot_hat = self.predictor.tau0 + phi @ w_hat

        # Force cost and acceleration constraint
        eq_constr = qddot_cmd - qddot_hat
        return self.objective_function(u, x[2]) + torch.dot(lamda, eq_constr)

    def adaptive_FOO_vector(self, t, u, lamda, x, x_hat, w_hat):
        dL_du, dL_dlamda = grad(self.adaptive_lagrangian, argnums=(1, 2))(
            t, u, lamda, x, x_hat, w_hat
        )
        return torch.cat((dL_du, dL_dlamda))

    def adaptive_u_ff(self, t, u, lamda, x, x_hat, w_hat, adaptation):
        # Optimality vector and its derivatives
        foo = self.adaptive_FOO_vector(t, u, lamda, x, x_hat, w_hat)
        dFOO_dt, dFOO_dx, dFOO_dx_hat, dFOO_dw_hat = jacfwd(
            self.adaptive_FOO_vector, argnums=(0, 3, 4, 5)
        )(t, u, lamda, x, x_hat, w_hat)

        # Plant state derivative from the estimated model
        phi = self.predictor.regressor(x, u)
        qddot_hat = self.predictor.tau0 + phi @ w_hat
        x_dot_model = self.predictor.A_x @ x + self.predictor.B_x @ qddot_hat

        # Predicted state and parameter derivatives
        x_hat_dot = self.predictor.run_dynamics(x, u, w_hat, x_hat=x_hat)
        w_hat_dot = adaptation.run_dynamics(x - x_hat, phi, foo, dFOO_dx, w_hat)

        # Adaptive feedforward term
        uff = (
            dFOO_dt
            + dFOO_dx @ x_dot_model
            + dFOO_dx_hat @ x_hat_dot
            + dFOO_dw_hat @ w_hat_dot
        )
        return uff, foo, x_hat_dot, w_hat_dot

    def adaptive_allocator_dynamics(self, t, u, lamda, x, x_hat, w_hat, adaptation):
        uff, foo, x_hat_dot, w_hat_dot = self.adaptive_u_ff(
            t, u, lamda, x, x_hat, w_hat, adaptation
        )

        # Hessian of the adaptive Lagrangian
        dFOO_du, dFOO_dlamda = jacfwd(
            self.adaptive_FOO_vector, argnums=(1, 2)
        )(t, u, lamda, x, x_hat, w_hat)
        hessian = torch.cat((dFOO_du, dFOO_dlamda), dim=1)

        y_dot = torch.linalg.solve(hessian, -(self.Gamma @ foo + uff))
        return y_dot, x_hat_dot, w_hat_dot

    def adaptive_allocate(self, t, x, x_dot, dt, adaptation):
        u = self.u
        lamda = self.lamda
        x_hat = self.predictor.state
        w_hat = adaptation.w_hat
        n_u = 2 * self.N
        half_dt = dt / 2

        # First stage and midpoint values
        k1, x_hat_dot, w_hat_dot = self.adaptive_allocator_dynamics(
            t, u, lamda, x, x_hat, w_hat, adaptation
        )
        u_mid = u + half_dt * k1[:n_u]
        lamda_mid = lamda + half_dt * k1[n_u:]
        x_mid = x + half_dt * x_dot
        x_hat_mid = x_hat + half_dt * x_hat_dot
        w_hat_mid = w_hat + half_dt * w_hat_dot

        # Evaluate the derivatives again at the midpoint
        k2, x_hat_dot, w_hat_dot = self.adaptive_allocator_dynamics(
            t + half_dt, u_mid, lamda_mid, x_mid, x_hat_mid, w_hat_mid, adaptation
        )

        self.u = u + dt * k2[:n_u]
        self.lamda = lamda + dt * k2[n_u:]
        self.predictor.state = x_hat + dt * x_hat_dot
        adaptation.w_hat = w_hat + dt * w_hat_dot

        return self.u





