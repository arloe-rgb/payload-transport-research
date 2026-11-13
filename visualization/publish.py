"""Prepare a static GitHub Pages checkout from selected replay exports."""

import argparse
import json
import math
import re
from pathlib import Path
from shutil import copyfile


SOURCE = Path(__file__).resolve().parent
ASSETS = (
    "index.html", "style.css", "player.js", "scene.js",
    "vendor/three-replay.min.js", "vendor/THREE-LICENSE.txt",
)
ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
REQUIRED = {"version", "payload", "forceLimits", "time", "state", "reference", "u"}
OPTIONAL = {"lambda", "gradientU", "gradientLambda"}
POSITION_HISTORIES = {"attachment_positions", "uav_positions"}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value, size):
    return isinstance(value, list) and len(value) == size and all(map(finite, value))


def read_replay(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not finite(data.get("version")) or data["version"] != 1:
        raise ValueError("Expected a version 1 replay export.")
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f"Missing replay fields: {', '.join(sorted(missing))}.")
    unknown = data.keys() - REQUIRED - OPTIONAL - POSITION_HISTORIES
    if unknown:
        raise ValueError(f"Unexpected replay fields: {', '.join(sorted(unknown))}.")
    payload = data["payload"]
    if not isinstance(payload, dict) or payload.keys() != {"width", "height", "attachments"}:
        raise ValueError("Invalid payload geometry fields.")
    attachments = payload["attachments"]
    if (not isinstance(attachments, list) or not attachments
            or not all(vector(row, 2) for row in attachments)
            or not all(finite(payload[key]) and payload[key] > 0 for key in ("width", "height"))):
        raise ValueError("Invalid payload geometry.")
    count = len(attachments)
    limits = data["forceLimits"]
    if not vector(limits, count) or min(limits) <= 0:
        raise ValueError("Force limits must be positive, one per UAV.")
    times = data["time"]
    if (not isinstance(times, list) or not times or not all(map(finite, times))
            or any(a >= b for a, b in zip(times, times[1:]))):
        raise ValueError("Expected finite, increasing timestamps.")
    for key, width in (
        ("state", 6), ("reference", 6), ("u", 2 * count),
        ("lambda", 3), ("gradientU", 2 * count), ("gradientLambda", 3),
    ):
        if key not in data:
            continue
        rows = data[key]
        if (not isinstance(rows, list) or len(rows) != len(times)
                or not all(vector(row, width) for row in rows)):
            raise ValueError(f"Invalid history: {key}.")
    for key in sorted(POSITION_HISTORIES & data.keys()):
        rows = data[key]
        allow_undefined = key == "uav_positions"
        if (not isinstance(rows, list) or len(rows) != len(times)
                or not all(
                    isinstance(row, list) and len(row) == count
                    and all(vector(point, 2) or (allow_undefined and point == [None, None])
                            for point in row)
                    for row in rows
                )):
            raise ValueError(f"Invalid history: {key}.")
    return raw, {
        "duration": times[-1] - times[0],
        "metadata": {"uavs": count, "samples": len(times), "replayVersion": 1},
    }


def checked_path(checkout, relative):
    path = checkout / relative
    if path.resolve() != path:
        raise ValueError(f"Refusing to write through a link: {path}")
    return path


def sync(checkout):
    files = {f"site/{name}": SOURCE / name for name in ASSETS}
    files.update({
        "README.md": SOURCE / "pages" / "README.md",
        ".github/workflows/deploy.yml": SOURCE / "pages" / "deploy.yml",
    })
    for relative, source in files.items():
        target = checked_path(checkout, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)


def add_run(checkout, args):
    if not ID_PATTERN.fullmatch(args.id):
        raise ValueError("Use a lowercase ID such as circular-baseline.")
    raw, summary = read_replay(args.replay)
    manifest_path = checked_path(checkout, "site/datasets/manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {"version": 1, "datasets": []}
    )
    if manifest.get("version") != 1:
        raise ValueError("Unsupported dataset manifest version.")
    file = f"datasets/{args.id}.json"
    target = checked_path(checkout, f"site/{file}")
    if target.exists() or any(run["id"] == args.id for run in manifest["datasets"]):
        raise ValueError(f"Dataset ID already exists: {args.id}")
    entry = {"id": args.id, "name": args.name, "file": file, **summary}
    if args.description:
        entry["description"] = args.description
    manifest["datasets"].append(entry)
    if args.default or not manifest.get("default"):
        manifest["default"] = args.id
    encoded = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if not args.no_sync:
        sync(checkout)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    temporary = manifest_path.with_suffix(".json.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        temporary.replace(manifest_path)
    except OSError:
        target.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Separate public checkout directory.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sync", help="Copy the viewer and deployment files only.")
    add = commands.add_parser("add", help="Add one selected export and sync the viewer.")
    add.add_argument("replay", type=Path)
    add.add_argument("--id", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--description")
    add.add_argument("--default", action="store_true")
    add.add_argument("--no-sync", action="store_true", help="Add the dataset without copying viewer assets.")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    research = SOURCE.parent
    if checkout == research or checkout.is_relative_to(research) and not checkout.is_relative_to(research / "outputs"):
        parser.error("Use a separate checkout or a staging directory under outputs/.")
    try:
        if args.command == "add":
            add_run(checkout, args)
        else:
            sync(checkout)
    except (ValueError, OSError, OverflowError) as error:
        parser.error(str(error))
    print(f"Prepared static site in {checkout / 'site'}. Review, commit, and push the public checkout.")


if __name__ == "__main__":
    main()
