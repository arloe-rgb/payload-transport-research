import torch

class UAV:

    def __init__(self, config):

        payload = config["payload"]

        self.N = payload["N"]

        self.Lengths = torch.tensor(payload["cable_lengths"], dtype = torch.float64)

        self.Pc_B = torch.tensor(payload["attachment_points"], dtype = torch.float64)

    def attachment_points(self, x):

        # Calculate Rotation Matrix
        Pc_B = self.Pc_B.to(x)
        theta = x[2]
        R = torch.stack((torch.stack((torch.cos(theta), -torch.sin(theta))),
                                 torch.stack((torch.sin(theta), torch.cos(theta)))))

        # Cable Positions in the Inertial Frame
        Pc_W = Pc_B @ R.T + x[:2]

        return Pc_W

    def UAV_positions(self, x, u_opt):
        """Reconstruct cable endpoints, with NaN coordinates for zero-force rows."""

        # Attachment points in the world frame
        Pc_W = self.attachment_points(x)

        u = u_opt.reshape(self.N, 2).to(device=x.device)
        L = self.Lengths.to(x)[:, None]

        # Scale before taking the norm to preserve directions at extreme magnitudes.
        scale = u.abs().amax(dim=1, keepdim=True)
        scaled_u = u / scale.masked_fill(scale == 0, torch.nan)
        u_dir = scaled_u / torch.linalg.norm(scaled_u, dim=1, keepdim=True)
        uav_positions = Pc_W + (L * u_dir)

        return uav_positions
