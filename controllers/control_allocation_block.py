import numpy as np
import yaml
import torch
from torch.func import grad, jacfwd, hessian

class allocator:

    def __init__(self, config):

        uav = config["UAV"]

        # Number of UAVs
        self.N = uav["N"]

        # Init decison variable u
        self.u = torch.zeros(self.N*2)

        # multipliers for equality constraints
        self.lamda = torch.zeros(3)

        # log barrier scalar
        self.log_scalar = uav["log_scalar"]

        # Optimization weights (one per UAV)
        self.weights = torch.tensor(uav["weights"], dtype=torch.float32)

        # Maximum force for each UAV
        self.F_max = torch.tensor(uav["F_max"], dtype=torch.float32)

    def objective_function(self, u_control):

        # reshape u for easier computation
        u = u_control.reshape(self.N, 2)

        # Calc (||u_i||_2)^2 first
        force_mag = torch.sum(u**2, dim=1)

        # 1/2 s^T * ||u||^2
        quad_cost = 0.5 * torch.dot(self.weights, force_mag)

        # Inequality constraints 
        force_mag_constr = force_mag - self.F_max**2   # Max force constraints
        force_dir_constr = -u[:, 1]   # Force direction constraints

        # Log barrier cost function
        barrier_cost = -self.log_scalar*(
            torch.sum(torch.log(-force_mag_constr))
            + torch.sum(torch.log(-force_dir_constr))
        )

        return quad_cost + barrier_cost

    def equality_constraints(self, B, u, tau_d):

        equal_constr = B @ u - tau_d

        return equal_constr

    def lagrangian(self, t, u, lamda, B, tau_d):

        obj_func = self.objective_function(u)

        eq_constr = self.equality_constraints(B, u, tau_d)

        L = obj_func + torch.dot(lamda, eq_constr)

        return L

    def FOO_vector(self, t, u, lamda, B, tau_d):

        dL_du, dL_dlamda = grad(self.lagrangian, argnums=(1, 2))(t, u, lamda, B, tau_d)

        return torch.cat((dL_du, dL_dlamda))

    def hessian(self, t, u, lamda, B, tau_d):

        dFOO_du, dFOO_dlamda = jacfwd(self.FOO_vector, argnums=(1, 2))(t, u, lamda, B, tau_d)

        return torch.cat((dFOO_du, dFOO_dlamda), dim=1)

    def u_ff(self, t, u, lamda, B, tau_d, chi, chi_dot):

        # time derivative of FOO_vector
        dFOO_dt = jacfwd(self.FOO_vector, argnums=0)(t, u, lamda, B, tau_d, chi, chi_dot)

        # state derivative of FOO_vector
        dFOO_dchi = jacfwd(self.FOO_vector, argnums=5)(t, u, lamda, B, tau_d, chi, chi_dot)

        uff = dFOO_dt + dFOO_dchi @ chi_dot

        return uff

    def allocator_dynamics(self, t, u, lamda, B, tau_d, chi, chi_dot):

        # Convert inputs to torch tensors
        chi = torch.as_tensor(chi, dtype=torch.float64)
        chi_dot = torch.as_tensor(chi_dot, dtype=torch.float64)
        u = torch.as_tensor(u, dtype=torch.float64)
        tau_d = torch.as_tensor(tau_d, dtype=torch.float64)

        FOO_vector = self.FOO_vector(t, u, lamda, B, tau_d)

        hessian = self.hessian(t, u, lamda, B, tau_d)

        uff = self.u_ff(t, u, lamda, B, tau_d, chi, chi_dot)

        ff_term = self.Gamma @ FOO_vector + uff

        # constructing the dynamics of the decision variables and multipliers
        ud_ld = torch.linalg.solve(hessian, -ff_term)

        return ud_ld


    def integrate()



