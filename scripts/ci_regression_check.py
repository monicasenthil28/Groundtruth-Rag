"""Exercise regression CLI exit codes against committed synthetic fixtures."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/regression_ci"
OUTPUT = ROOT / "reports/ci"


def main() -> int:
    cases = (
        ("pass", "PASS", 0, False),
        ("warn", "WARN", 0, False),
        ("warn_strict", "WARN", 1, True),
        ("fail", "FAIL", 1, False),
    )
    lines = ["## Synthetic Regression Fixture Gate", "",
             "These values are synthetic CLI checks, not OSHA benchmark evidence.", ""]
    for name, expected_status, expected_code, strict in cases:
        candidate = "warn" if name == "warn_strict" else name
        directory = OUTPUT / name
        for filename in ("regression.json", "REGRESSION.md"):
            (directory / filename).unlink(missing_ok=True)
        command = [
            sys.executable, "-m", "groundtruth_rag.cli", "regress",
            "--baseline", str(FIXTURES / "baseline.json"),
            "--candidate", str(FIXTURES / f"candidate_{candidate}.json"),
            "--policy", str(FIXTURES / "policy.yaml"),
            "--output", str(directory),
        ]
        if strict:
            command.append("--fail-on-warn")
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        report_path = directory / "regression.json"
        actual_status = None
        if report_path.exists():
            actual_status = json.loads(report_path.read_text(encoding="utf-8"))["overall_status"]
        if result.returncode != expected_code or actual_status != expected_status:
            raise RuntimeError(
                f"{name}: expected {expected_status}/exit {expected_code}, "
                f"got {actual_status}/exit {result.returncode}"
            )
        lines.append(f"- {name}: {actual_status} (exit {result.returncode}, expected)")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "fixture-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Synthetic regression fixture gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
