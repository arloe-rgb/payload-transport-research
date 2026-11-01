import numpy as np

class plant:
    def __int__(self, config, x_init):

        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["payload"]

        # Physical parameters
        self.m = config["mass"]
        self.J = config["inertia"]
        self.g = config["gravity"]

        # Mass matrix and gravity vector
        self.M = np.diag([self.m, self.m, self.J])
        self.M_inv = np.linalg.inv(self.M)
        self.G = np.array([0.0, -self.m*self.g, 0.0])

        # Constant attachment points (body frame)
        self.P_c_B = np.array(
            config["attachment_points"],
            dtype=float
        )

        # Number of UAVs
        self.N = self.P_c_B.shape[0]

        # Initial plant state
        self.state = np.array(x_init[:3], dtype=float)
        self.dstate = np.array(x_init[3:], dtype=float)

    def run_dynamics(self, u_optimal):

        # Calculate optimal desired wrench
        tau_d_opt =

    def B_mat(self):
