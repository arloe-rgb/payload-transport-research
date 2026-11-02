import numpy as np
import yaml

class reference_trajectory:

    def __init__(self, config):

        traj = config["trajectory"]

        self.type = traj["type"]
        self.duration = traj["duration"]

        self.q_start = np.array(traj["start"], dtype=float)
        self.q_goal = np.array(traj["goal"], dtype=float)

        if self.type == "circular" :
            self.center = np.array(traj["center"], dtype=float)
            self.radius = traj["radius"]
            self.omega = traj["omega"]
            self.theta_ref = traj["theta_ref"]

    def get_trajectory(self, t):

        if self.type == "linear":
            q_ref, qdot_ref, qddot_ref = self.linear(t)

        elif self.type == "circular":
            q_ref, qdot_ref, qddot_ref = self.circular(t)

        x_ref = np.concatenate((q_ref, qdot_ref))

        return x_ref, qddot_ref

    def linear_traj(self, t):
        pass

    def circular_traj(self, t):

        # STARTING WITH CONSTANT ORIENTATION REFERNCE ANGLE CAN EXPAND TO CHANGING ORIENTATION LATER

        # Circular trajectory parameters
        xc, zc = self.center
        r = self.radius
        omega = self.omega
        theta = self.theta_ref

        # Position
        q = np.array([
            xc + r * np.cos(omega * t),
            zc + r * np.sin(omega * t),
            theta
        ])

        # Velocity
        qdot = np.array([
            -r * omega * np.sin(omega * t),
            r * omega * np.cos(omega * t),
            0.0
        ])

        # Acceleration
        qddot = np.array([
            -r * omega**2 * np.cos(omega * t),
            -r * omega**2 * np.sin(omega * t),
            0.0
        ])

        return q, qdot, qddot