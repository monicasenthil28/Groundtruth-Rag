"""Offline metadata integrity and synthetic CLI regression gates."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from groundtruth_rag.benchmark_setup import load_spec, verify_benchmark_package
from groundtruth_rag.cli import app
from groundtruth_rag.labeling.manifest import file_sha256


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/regression_ci"


@pytest.fixture
def committed_package(tmp_path: Path):
    spec = load_spec(ROOT / "benchmarks/osha/benchmark.yaml")
    paths = ("benchmarks/osha/benchmark.yaml", spec.golden.dataset,
             spec.golden.review_manifest, spec.baseline, *spec.retriever_configs.values())
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path, spec


def _edit_json(path: Path, edit) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_committed_package_integrity_without_external_inputs(committed_package, monkeypatch):
    root, spec = committed_package
    assert not (root / "datasets/raw/osha3120.pdf").exists()
    assert not (root / "datasets/osha/corpus.jsonl").exists()
    monkeypatch.setattr("groundtruth_rag.benchmark_setup.load_corpus",
                        lambda path: pytest.fail("corpus was accessed"))
    monkeypatch.setattr("groundtruth_rag.benchmark_setup.urlopen",
                        lambda url: pytest.fail("network was accessed"))
    assert verify_benchmark_package(root, spec) == 29


def test_changed_golden_hash_fails(committed_package):
    root, spec = committed_package
    dataset = root / spec.golden.dataset
    content = dataset.read_text(encoding="utf-8")
    dataset.write_text(content.replace("What is lockout/tagout?", "What is lockout?", 1),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="golden dataset hash mismatch"):
        verify_benchmark_package(root, spec)


def test_wrong_query_count_fails(committed_package):
    root, spec = committed_package
    dataset = root / spec.golden.dataset
    lines = dataset.read_text(encoding="utf-8").splitlines()
    dataset.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="golden query count mismatch"):
        verify_benchmark_package(root, spec)


def test_q013_added_to_eval_fails(committed_package):
    root, spec = committed_package
    dataset = root / spec.golden.dataset
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rows[12]["query_id"] = "q013"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    _edit_json(root / spec.golden.review_manifest,
               lambda data: data.update(dataset_sha256=file_sha256(dataset)))
    with pytest.raises(ValueError, match="skipped query in golden dataset"):
        verify_benchmark_package(root, spec)


def test_q013_removed_from_skipped_manifest_fails(committed_package):
    root, spec = committed_package
    _edit_json(root / spec.golden.review_manifest,
               lambda data: data.update(skipped_ids=["q999"]))
    with pytest.raises(ValueError, match="skipped IDs mismatch"):
        verify_benchmark_package(root, spec)


@pytest.mark.parametrize("key,value", [
    ("corpus_fingerprint", "wrong-fingerprint"),
    ("query_count", 28),
])
def test_baseline_metadata_mismatch_fails(committed_package, key, value):
    root, spec = committed_package
    _edit_json(root / spec.baseline, lambda data: data.update({key: value}))
    with pytest.raises(ValueError, match="baseline incompatible"):
        verify_benchmark_package(root, spec)


def test_missing_retriever_config_fails(committed_package):
    root, spec = committed_package
    (root / spec.retriever_configs["bm25"]).unlink()
    with pytest.raises(ValueError, match="cannot read config"):
        verify_benchmark_package(root, spec)


def test_cli_benchmark_verify_is_offline():
    result = CliRunner().invoke(app, ["benchmark-verify", "osha"])
    assert result.exit_code == 0, result.output
    assert "integrity: PASS" in result.output


@pytest.mark.parametrize("candidate,expected_status,expected_code,strict", [
    ("pass", "PASS", 0, False),
    ("warn", "WARN", 0, False),
    ("warn", "WARN", 1, True),
    ("fail", "FAIL", 1, False),
])
def test_synthetic_regression_cli(tmp_path: Path, candidate: str, expected_status: str,
                                  expected_code: int, strict: bool):
    output = tmp_path / "report"
    args = ["regress", "--baseline", str(FIXTURES / "baseline.json"),
            "--candidate", str(FIXTURES / f"candidate_{candidate}.json"),
            "--policy", str(FIXTURES / "policy.yaml"), "--output", str(output)]
    if strict:
        args.append("--fail-on-warn")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == expected_code, result.output
    report = json.loads((output / "regression.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == expected_status
    assert (output / "REGRESSION.md").exists()
