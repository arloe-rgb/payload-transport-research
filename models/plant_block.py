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
        self.Pc_B = np.array(payload["attachment_points"], dtype=float)

        # Number of UAVs
        self.N = sim["N"]

        # Initial plant state
        self.x_init = np.array(sim["initial_cond"], dtype=float)
        self.state = np.array(self.x_init, dtype=float).copy()
   

    def run_dynamics(self, x, u):

        # calc B matrtix
        B = self.B_mat(x[2])

        # Calculate optimal desired wrench
        tau_d_opt = B @ u

        q_ddot = self.M_inv @ (tau_d_opt + self.G)

        return np.concatenate((x[3:], q_ddot))

    def integrate(self, u_optimal, dt):

        x = self.state

        k1 = self.run_dynamics(x, u)
        k2 = self.run_dynamics(x + 0.5*dt*k1, u_optimal)
        k3 = self.run_dynamics(x + 0.5*dt*k2, u_optimal)
        k4 = self.run_dynamics(x + dt*k3, u_optimal)

        self.state = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        return self.state

    def B_mat(self, theta):

        # Calculate Rotation Matrix
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

        # Cable Positions in the Inertial Frame
        Pc_W = self.Pc_B @ R.T

        # Calc Allocation matrix
        B = np.zeros((3, 2*self.N))

        B[0, 0::2] = 1.0
        B[1, 1::2] = 1.0

        B[2, 0::2] = -Pc_W[:, 1]
        B[2, 1::2] =  Pc_W[:, 0]

        return B