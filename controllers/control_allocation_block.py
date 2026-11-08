import numpy as np
import yaml

class allocator:

    def __init__(self, config):

        uav = config["UAV"]

        # Number of UAVs
        self.N = uav["N"]

        # Init decison variable u make it N x 2 for easier computation
        self.u = np.zeros((self.N*2, 1))

        # Optimization weights (one per UAV)
        self.weights = np.array(uav["weights"], dtype=float)

        # Maximum force for each UAV
        self.F_max = np.array(uav["F_max"], dtype=float)

    def objective_function(self):

        # reshape u for easier computation
        u = self.u.reshape(self.N, 2)

        # Calc (||u_i||_2)^2 first
        force_mag = np.sum(u**2, axis=1)

        # can then combine force_mags with weights into the objective function
        obj_fun = 0.5 * np.dot(self.weights, force_mag)

        return obj_fun

    def equality_constraints(self, B, tau_d):

        equal_constr = B @ self.u - tau_d

        return equal_constr

    def inequality_constraints(self):

        # reshape self.u for easier computation
        u = self.u.reshape(self.N, 2)

        # Makes sure that each UAV does not exceed its respective max force
        force_mag_constr = self.F_max**2 - np.sum(u**2, axis=1)

        # Make sure that the vertical component for each force it always positive
        force_dir_constr = u[:,1]

        # make vector of all inequality constraint equations
        inequ_constr = np.concatenate((force_mag_constr, force_dir_constr))

        return inequ_constr

