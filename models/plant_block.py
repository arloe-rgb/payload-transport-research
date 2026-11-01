import numpy as np
import yaml


class plant:
    def __init__(self, config):

        payload = config["payload"]
        sim = config["SIM"]

        # Physical parameters
        self.m = payload["mass"]
        self.J = payload["inertia"]
        self.g = payload["gravity"]

        # Mass matrix and gravity vector
        self.M = np.diag([self.m, self.m, self.J])
        self.M_inv = np.linalg.inv(self.M)
        self.G = np.array([0.0, -self.m*self.g, 0.0])

        # Constant attachment points (body frame)
        self.Pc_B = np.array(config["attachment_points"], dtype=float)

        # Number of UAVs
        self.N = sim["N"]

        # Initial plant state
        self.x_init = np.array(config["initial_cond"], dtype=float)
        self.state = np.array(self.x_init[:3], dtype=float).copy()
        self.dstate = np.array(self.x_init[3:], dtype=float).copy()


    def run_dynamics(self, u_optimal, B):

        # Calculate optimal desired wrench
        tau_d_opt = B @ u_optimal

        x_ddot = self.M_inv @ (tau_d_opt + self.G)

        return x_ddot

    def B_mat(self):

        # Extract theta
        theta = self.state[2]

        # Calculate Rotation Matrix
        R = np.array([[cos(theta), -sin(theta)], 
                     [sin(theta), cos(theta)]])

        # Cable Positions in the Inertial Frame
        Pc_W = self.Pc_B @ R.T

        # Calc Allocation matrix
        B = np.zeros((3, 2*self.N))

        B[0, 0::2] = 1.0
        B[1, 1::2] = 1.0

        B[2, 0::2] = -Pc_W[:, 1]
        B[2, 1::2] =  Pc_W[:, 0]

        return B