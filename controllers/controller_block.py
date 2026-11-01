class PD_FF_Controller:

    def __init__(self, Kp, Kd, m, J):
        # defining gain matrices
        self.Kp = Kp
        self.Kd = Kd

        # Constant plant parameters (just make them attributes of object since const.)
        g = 9.81

        self.M = np.diag([m, m, J])
        self.G = np.array([0.0, -m*g, 0.0])

        # Defining error terms (may pass these into a object for calculating derivatives)
        self.prev_error = 0.0

    def control_cmd(self, x, x_d, xdot, xdot_d, xddot_r):
        # calc error
        e = x_d - x
        edot = xdot_d - xdot

        xddot_d = xddot_r + self.Kd @ edot + self.Kp @ e

        tau_d = (self.M @ xddot_d ) - self.G

        return tau_d
