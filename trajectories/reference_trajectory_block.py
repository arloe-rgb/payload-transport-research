import torch
import yaml

class reference_trajectory:

    def __init__(self, config):

        traj = config["reference_trajectory"]
        orientation = traj["orientation"]

        self.type = traj["type"]
        self.orientation_type = orientation["type"]
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

        if self.orientation_type == "sinusoidal":
                    self.theta_amplitude = orientation["amplitude"]
                    self.theta_omega = orientation["omega"]
                    self.theta_phase = orientation["phase"]

    def get_trajectory(self, t):

        pos_ref, vel_ref, acc_ref = self.position_traj(t)

        theta_ref, theta_dot_ref, theta_ddot_ref = self.orientation_traj(t)

        q_ref = torch.cat((pos_ref, theta_ref.reshape(1)))
        qdot_ref = torch.cat((vel_ref, theta_dot_ref.reshape(1)))
        qddot_ref = torch.cat((acc_ref, theta_ddot_ref.reshape(1)))

        x_ref = torch.cat((q_ref, qdot_ref))

        return x_ref, qddot_ref

    def position_traj(self, t):

        t = torch.as_tensor( t, dtype=self.q_start.dtype, device=self.q_start.device)

        if self.type == "circular":

            # Circular trajectory parameters
            xc, zc = self.center
            r = self.radius
            omega = self.omega

            # Position
            pos_ref = torch.stack([
                xc + r * torch.cos(omega * t),
                zc + r * torch.sin(omega * t)
            ])

            # Velocity
            vel_ref = torch.stack([
                -r * omega * torch.sin(omega * t),
                r * omega * torch.cos(omega * t)
            ])

            # Acceleration
            acc_ref = torch.stack([
                -r * omega**2 * torch.cos(omega * t),
                -r * omega**2 * torch.sin(omega * t)
            ])

        elif self.type == "sinusoidal":

            # Sinusoidal trajectory parameters
            A = self.amplitude
            omega = self.omega
            vx = self.vx

            x0, z0 = self.q_start[:2]

            # Position
            pos_ref = torch.stack([
                x0 + vx * t,
                z0 + A * torch.sin(omega * t)
            ])

            # Velocity
            vel_ref = torch.stack([
                vx * torch.ones_like(t),
                A * omega * torch.cos(omega * t)
            ])

            # Acceleration
            acc_ref = torch.stack([
                torch.zeros_like(t),
                -A * omega**2 * torch.sin(omega * t)
            ])

        else:
            raise ValueError(f"Unsupported position trajectory: {self.type}")

        return pos_ref, vel_ref, acc_ref

    def orientation_traj(self, t):

        t = torch.as_tensor(t, dtype=self.q_start.dtype, device=self.q_start.device)

        theta0 = self.q_start[2]

        if self.orientation_type == "sinusoidal":

            A = self.theta_amplitude
            omega = self.theta_omega
            phi = self.theta_phase

            # Orientation
            theta_ref = theta0 + A * torch.sin(omega * t + phi)

            # Angular velocity
            theta_dot_ref = A * omega * torch.cos(omega * t + phi)

            # Angular acceleration
            theta_ddot_ref = -A * omega**2 * torch.sin(omega * t + phi)

        elif self.orientation_type == "constant":

            theta_ref = theta0 + torch.zeros_like(t)
            theta_dot_ref = torch.zeros_like(t)
            theta_ddot_ref = torch.zeros_like(t)

        else:
            raise ValueError(
                f"Unsupported orientation trajectory: {self.orientation_type}"
            )

        return theta_ref, theta_dot_ref, theta_ddot_ref

    # def linear_traj(self, t):
        #     pass
    
        # def circular_traj(self, t):
    
        #     # STARTING WITH CONSTANT ORIENTATION REFERNCE ANGLE CAN EXPAND TO CHANGING ORIENTATION LATER
    
        #     # Circular trajectory parameters
        #     xc, zc = self.center
        #     r = self.radius
        #     omega = self.omega
        #     theta = torch.as_tensor(self.theta_ref, dtype=self.center.dtype, device=self.center.device)
        #     t = torch.as_tensor(t, dtype=self.center.dtype, device=self.center.device)
    
        #     # Position
        #     q = torch.stack([
        #         xc + r * torch.cos(omega * t),
        #         zc + r * torch.sin(omega * t),
        #         theta
        #     ])
    
        #     # Velocity
        #     qdot = torch.stack([
        #         -r * omega * torch.sin(omega * t),
        #         r * omega * torch.cos(omega * t),
        #         torch.zeros_like(t)
        #     ])
    
        #     # Acceleration
        #     qddot = torch.stack([
        #         -r * omega**2 * torch.cos(omega * t),
        #         -r * omega**2 * torch.sin(omega * t),
        #         torch.zeros_like(t)
        #     ])
    
        #     return q, qdot, qddot
    
        # def sinusoidal_traj(self, t):
    
        #     # Sinusoidal trajectory parameters
        #     A = self.amplitude
        #     omega = self.omega
        #     vx = self.vx
    
        #     x0, z0, theta = self.q_start
    
        #     t = torch.as_tensor(
        #         t,
        #         dtype=self.q_start.dtype,
        #         device=self.q_start.device
        #     )
    
        #     # Position
        #     q = torch.stack([
        #         x0 + vx * t,
        #         z0 + A * torch.sin(omega * t),
        #         theta
        #     ])
    
        #     # Velocity
        #     qdot = torch.stack([
        #         vx * torch.ones_like(t),
        #         A * omega * torch.cos(omega * t),
        #         torch.zeros_like(t)
        #     ])
    
        #     # Acceleration
        #     qddot = torch.stack([
        #         torch.zeros_like(t),
        #         -A * omega**2 * torch.sin(omega * t),
        #         torch.zeros_like(t)
        #     ])
    
        #     return q, qdot, qddot
