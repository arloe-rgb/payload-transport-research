"""Serialize a completed simulation for the local browser player."""

import json
from math import isfinite
from pathlib import Path


def write_replay(
    path, payload, limits, time, state, reference, forces, multipliers, gradients,
    *, attachment_positions=None, uav_positions=None,
):
    """Write version 1 replay data with optional world-frame position histories.

    Position histories are indexed by time, UAV, and [x, z]. Undefined UAV
    positions are stored as [null, null].
    """
    def values(tensor):
        return tensor.detach().cpu().tolist()

    num_forces = forces.shape[1]
    replay = {
        "version": 1,
        "payload": {
            "width": payload["width"],
            "height": payload["height"],
            "attachments": payload["attachment_points"],
        },
        "forceLimits": values(limits),
        "time": values(time),
        "state": values(state),
        "reference": values(reference),
        "u": values(forces),
        "lambda": values(multipliers),
        "gradientU": values(gradients[:, :num_forces]),
        "gradientLambda": values(gradients[:, num_forces:]),
    }
    if attachment_positions is not None:
        replay["attachment_positions"] = values(attachment_positions)
    if uav_positions is not None:
        replay["uav_positions"] = [
            [[x, z] if isfinite(x) and isfinite(z) else [None, None] for x, z in sample]
            for sample in values(uav_positions)
        ]
    encoded = json.dumps(replay, allow_nan=False, separators=(",", ":"))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
