import numpy as np
import yaml

class allocator:

    def __init__(self, config):

        uav = config["UAV"]

        # Number of UAVs
        self.N = uav["N"]

        # Init decison variable u
        self.u = np.zeros(self.N*2)

        # multipliers for equality constraints
        self.q = np.zeros(3)

        # log barrier scalar
        self.log_scalar = uav["log_scalar"]

        # Optimization weights (one per UAV)
        self.weights = np.array(uav["weights"], dtype=float)

        # Maximum force for each UAV
        self.F_max = np.array(uav["F_max"], dtype=float)

    def objective_function(self, u_control):

        # reshape u for easier computation
        u = u_control.reshape(self.N, 2)

        # Calc (||u_i||_2)^2 first
        force_mag = np.sum(u**2, axis=1)

        # 1/2 s^T * ||u||^2
        quad_cost = 0.5 * np.dot(self.weights, force_mag)

        # Inequality constraints 
        force_mag_constr = force_mag - self.F_max**2   # Max force constraints
        force_dir_constr = -u[:, 1]   # Force direction constraints

        # Log barrier cost function
        barrier_cost = -self.log_scalar*(
            np.sum(np.log(-force_mag_constr))
            + np.sum(np.log(-force_dir_constr))
        )

        return quad_cost + barrier_cost

    def equality_constraints(self, B, u, tau_d):

        equal_constr = B @ u - tau_d

        return equal_constr

    # def inequality_constraints(self):

    #     # reshape self.u for easier computation
    #     u = self.u.reshape(self.N, 2)

    #     # Makes sure that each UAV does not exceed its respective max force
    #     force_mag_constr = np.sum(u**2, axis=1) - self.F_max**2

    #     # Make sure that the vertical component for each force it always positive
    #     force_dir_constr = -u[:,1]

    #     # make vector of all inequality constraint equations
    #     inequ_constr = np.concatenate((force_mag_constr, force_dir_constr))

    #     return inequ_constr

    def lagrangian(self, u, q, B, tau_d):

        obj_func = self.objective_function(u)

        eq_constr = self.equality_constraints(B, u, tau_d)

        L = obj_func + np.dot(q, eq_constr)

        return L



