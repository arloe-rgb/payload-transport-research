"""Serialize a completed simulation for the local browser player."""

import json
from pathlib import Path


def write_replay(path, payload, limits, time, state, reference, forces, multipliers, gradients):
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
    encoded = json.dumps(replay, allow_nan=False, separators=(",", ":"))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
