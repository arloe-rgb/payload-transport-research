import torch
import yaml

class reference_trajectory:

    def __init__(self, config):

        traj = config["reference_trajectory"]

        self.type = traj["type"]
        self.duration = traj["duration"]

        self.q_start = torch.tensor(traj["start"], dtype=torch.float64, device="cpu")
        self.q_goal = self.q_start.new_tensor(traj["goal"])
        self.theta_ref = self.q_start[2]

        if self.type == "circular" :
            self.center = self.q_start.new_tensor(traj["circular"]["center"])
            self.radius = traj["circular"]["radius"]
            self.omega = traj["circular"]["omega"]

        if self.type == "sinusoidal":
            self.amplitude = traj["sinusoidal"]["amplitude"]
            self.omega = traj["sinusoidal"]["omega"]
            self.vx = traj["sinusoidal"]["vx"]

    def get_trajectory(self, t):

        if self.type == "linear":
            q_ref, qdot_ref, qddot_ref = self.linear(t)

        elif self.type == "circular":
            q_ref, qdot_ref, qddot_ref = self.circular_traj(t)

        elif self.type == "sinusoidal":
            q_ref, qdot_ref, qddot_ref = self.sinusoidal_traj(t)

        x_ref = torch.cat((q_ref, qdot_ref))

        return x_ref, qddot_ref

    def linear_traj(self, t):
        pass

    def circular_traj(self, t):

        # STARTING WITH CONSTANT ORIENTATION REFERNCE ANGLE CAN EXPAND TO CHANGING ORIENTATION LATER

        # Circular trajectory parameters
        xc, zc = self.center
        r = self.radius
        omega = self.omega
        theta = torch.as_tensor(self.theta_ref, dtype=self.center.dtype, device=self.center.device)
        t = torch.as_tensor(t, dtype=self.center.dtype, device=self.center.device)

        # Position
        q = torch.stack([
            xc + r * torch.cos(omega * t),
            zc + r * torch.sin(omega * t),
            theta
        ])

        # Velocity
        qdot = torch.stack([
            -r * omega * torch.sin(omega * t),
            r * omega * torch.cos(omega * t),
            torch.zeros_like(t)
        ])

        # Acceleration
        qddot = torch.stack([
            -r * omega**2 * torch.cos(omega * t),
            -r * omega**2 * torch.sin(omega * t),
            torch.zeros_like(t)
        ])

        return q, qdot, qddot

    def sinusoidal_traj(self, t):
        
        # Sinusoidal trajectory parameters
        A = self.amplitude
        omega = self.omega
        vx = self.vx

        x0, z0, theta = self.q_start

        t = torch.as_tensor(
            t,
            dtype=self.q_start.dtype,
            device=self.q_start.device
        )

        # Position
        q = torch.stack([
            x0 + vx * t,
            z0 + A * torch.sin(omega * t),
            theta
        ])

        # Velocity
        qdot = torch.stack([
            vx * torch.ones_like(t),
            A * omega * torch.cos(omega * t),
            torch.zeros_like(t)
        ])

        # Acceleration
        qddot = torch.stack([
            torch.zeros_like(t),
            -A * omega**2 * torch.sin(omega * t),
            torch.zeros_like(t)
        ])

        return q, qdot, qddot

