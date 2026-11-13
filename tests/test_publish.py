import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

from visualization.publish import read_replay


ROOT = Path(__file__).resolve().parents[1]


class PublishReplayTests(unittest.TestCase):
    def setUp(self):
        outputs = ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        directory = self.enterContext(TemporaryDirectory(dir=outputs))
        self.work = Path(directory)
        self.replay = self.work / "replay.json"
        self.checkout = self.work / "public"

    def publish(self, *options):
        return subprocess.run(
            [
                sys.executable, "visualization/publish.py", str(self.checkout), "add",
                str(self.replay), "--id", "test-run", "--name", "Test run", *options,
            ],
            cwd=ROOT, capture_output=True, text=True,
        )

    def test_generated_replay_publishes_with_position_histories(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        config["SIM"]["duration"] = config["SIM"]["dt"]
        config_path = self.work / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_simulation.py"), "--export-replay", str(self.replay)],
            cwd=self.work, env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        viewer = self.checkout / "site" / "player.js"
        viewer.parent.mkdir(parents=True)
        viewer.write_text("// existing viewer", encoding="utf-8")
        result = self.publish("--no-sync")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(viewer.read_text(encoding="utf-8"), "// existing viewer")

        datasets = self.checkout / "site" / "datasets"
        published = (datasets / "test-run.json").read_bytes()
        self.assertEqual(published, self.replay.read_bytes())
        data = json.loads(published)
        for key in ("attachment_positions", "uav_positions"):
            self.assertEqual(len(data[key]), 2)
            self.assertTrue(all(len(row) == config["payload"]["N"] for row in data[key]))
        manifest = json.loads((datasets / "manifest.json").read_bytes())
        self.assertEqual(manifest["datasets"][0]["file"], "datasets/test-run.json")
        self.assertEqual(manifest["datasets"][0]["metadata"]["replayVersion"], 1)

    def legacy_replay(self):
        return {
            "version": 1,
            "payload": {"width": 4, "height": 2, "attachments": [[0, 1]]},
            "forceLimits": [10],
            "time": [0],
            "state": [[0] * 6],
            "reference": [[0] * 6],
            "u": [[0, 0]],
        }

    def test_legacy_replay_publishes(self):
        self.replay.write_text(json.dumps(self.legacy_replay()), encoding="utf-8")
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.checkout / "site" / "player.js").is_file())

    def test_undefined_uav_position_is_valid(self):
        replay = self.legacy_replay() | {
            "attachment_positions": [[[0, 1]]],
            "uav_positions": [[[None, None]]],
        }
        self.replay.write_text(json.dumps(replay), encoding="utf-8")
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        published = self.checkout / "site" / "datasets" / "test-run.json"
        self.assertEqual(json.loads(published.read_bytes()), replay)

    def test_invalid_replays_are_rejected(self):
        legacy = self.legacy_replay()
        cases = [
            ("version 1", legacy | {"version": 2}),
            ("Missing replay fields: state", {k: v for k, v in legacy.items() if k != "state"}),
            ("Unexpected replay fields: researchDiagnostics", legacy | {"researchDiagnostics": {}}),
            ("Invalid history: u", legacy | {"u": [[0]]}),
            ("Invalid history: attachment_positions", legacy | {"attachment_positions": []}),
            ("Invalid history: attachment_positions", legacy | {"attachment_positions": [[[None, None]]]}),
            ("Invalid history: uav_positions", legacy | {"uav_positions": [[]]}),
            ("Invalid history: uav_positions", legacy | {"uav_positions": [[[0]]]}),
            ("Invalid history: uav_positions", legacy | {"uav_positions": [[[None, 0]]]}),
            ("Invalid history: uav_positions", legacy | {"uav_positions": [[[float("nan"), 0]]]}),
            ("Invalid history: uav_positions", legacy | {"uav_positions": [[[float("inf"), 0]]]}),
        ]
        for message, replay in cases:
            with self.subTest(message=message, replay=replay):
                self.replay.write_text(json.dumps(replay), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_replay(self.replay)
        result = self.publish()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.checkout.exists())
