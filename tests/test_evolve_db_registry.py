import importlib.util
import json
from pathlib import Path
import shutil
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evolve_db_registry.py"
SPEC = importlib.util.spec_from_file_location("evolve_db_registry", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REGISTRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REGISTRY
SPEC.loader.exec_module(REGISTRY)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_checkpoint(
    root: Path,
    *,
    task: str = "symbolic_accuracy_judge",
    subtask: str = "symbolic_accuracy_judge",
    model: str = "openai/gpt-oss-120b",
    policy: str = "rpucg",
    instance_name: str = "instance-1532096c",
    checkpoint_name: str = "db_state_154226",
    best_score: float = 1644854272.0,
    completed_evaluations: int = 105,
    inferable_paths: bool = True,
) -> dict[str, Path]:
    date_dir = "2026-03-23"
    model_dir = model.replace("/", "_")
    instance_dir = (
        root
        / "checkpoints"
        / date_dir
        / task
        / subtask
        / model_dir
        / "single"
        / date_dir
        / instance_name
    )
    checkpoint_dir = instance_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if inferable_paths:
        init_program = f"datasets/{task}/{subtask}/init_program.py"
        evaluator_path = f"datasets/{task}/{subtask}/evaluator.py"
        instruction_path = f"datasets/{task}/{subtask}/{subtask}.txt"
    else:
        init_program = "init.py"
        evaluator_path = "evaluator.py"
        instruction_path = "instruction.txt"

    _write_json(
        checkpoint_dir / "metadata.json",
        {
            "instance_id": instance_name.replace("instance-", ""),
            "completed_evaluations": completed_evaluations,
            "generation_attempts": completed_evaluations - 1,
            "generation_failures": 0,
            "generation_cancellations": 0,
            "evaluation_failures": 0,
            "best_score": best_score,
            "best_node_id": "best-node",
        },
    )
    _write_json(
        checkpoint_dir / "config.json",
        {
            "init_program": init_program,
            "evaluator_path": evaluator_path,
            "instruction_path": instruction_path,
            "output_path": str(instance_dir.parent),
            "model": model,
            "selector": policy,
        },
    )
    _write_json(checkpoint_dir / "policy.json", {"name": policy, "state": {}})
    _write_json(
        checkpoint_dir / "nodes.json",
        [{"id": "best-node", "score": best_score, "created_at": "2026-03-23T00:00:00Z"}],
    )
    _write_json(checkpoint_dir / "reflection.json", [])
    _write_json(checkpoint_dir / "failure.json", [])
    (checkpoint_dir / "best_program.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
    (checkpoint_dir / f"scores_{completed_evaluations:06d}.csv").write_text(
        "id,score,created_at,parent_ids,gen_id,chain_idx,error\nroot,1.0,2026-03-23T00:00:00Z,,0,0,\n",
        encoding="utf-8",
    )
    (checkpoint_dir / f"scores_{completed_evaluations:06d}.png").write_bytes(b"png")
    (checkpoint_dir / f"quantile_plot_{completed_evaluations:06d}.png").write_bytes(b"png")
    (instance_dir / "run.log").write_text("SimpleTES test run\n", encoding="utf-8")
    return {
        "instance_dir": instance_dir,
        "checkpoint_dir": checkpoint_dir,
    }


def test_expand_inputs_supports_latest_and_all(tmp_path):
    one = _make_checkpoint(
        tmp_path,
        instance_name="instance-shared",
        checkpoint_name="db_state_120000",
        completed_evaluations=10,
    )
    two = _make_checkpoint(
        tmp_path,
        instance_name="instance-shared",
        checkpoint_name="db_state_130000",
        completed_evaluations=20,
    )

    latest = REGISTRY.expand_inputs([str(one["instance_dir"])], select="latest")
    assert [path.name for path in latest] == [two["checkpoint_dir"].name]

    all_paths = REGISTRY.expand_inputs([str(one["instance_dir"])], select="all")
    assert [path.name for path in all_paths] == [one["checkpoint_dir"].name, two["checkpoint_dir"].name]


def test_commit_creates_symlinks_registry_and_dashboard(tmp_path):
    first = _make_checkpoint(
        tmp_path,
        task="symbolic_accuracy_judge",
        subtask="symbolic_accuracy_judge",
        instance_name="instance-1532096c",
        checkpoint_name="db_state_154226",
        best_score=1644854272.0,
    )
    second = _make_checkpoint(
        tmp_path,
        task="autocorrelation",
        subtask="autocorrelation_second",
        instance_name="instance-8eabcb1d",
        checkpoint_name="db_state_164728",
        best_score=42.0,
        policy="balance",
    )
    shared_root = tmp_path / "shared"

    result = REGISTRY.main(
        [
            "--shared-root",
            str(shared_root),
            "commit",
            "--name",
            "Alice Example",
            "--notes",
            "two checkpoints from the same sweep",
            "--job-id",
            "job-1001",
            "--job-id",
            "job-1002",
            "--tag",
            "nightly",
            str(first["checkpoint_dir"]),
            str(second["checkpoint_dir"]),
        ]
    )

    assert result == 0
    entry_manifests = sorted((shared_root / "registry" / "entries").glob("*.json"))
    batch_manifests = sorted((shared_root / "registry" / "batches").glob("*.json"))
    assert len(entry_manifests) == 2
    assert len(batch_manifests) == 1

    first_manifest = json.loads(entry_manifests[0].read_text(encoding="utf-8"))
    second_manifest = json.loads(entry_manifests[1].read_text(encoding="utf-8"))
    assert first_manifest["notes"] == "two checkpoints from the same sweep"
    assert first_manifest["job_ids"] == ["job-1001", "job-1002"]
    assert first_manifest["tags"] == ["nightly"]
    assert {first_manifest["subtask"], second_manifest["subtask"]} == {
        "symbolic_accuracy_judge",
        "autocorrelation_second",
    }

    link_paths = [shared_root / first_manifest["shared"]["link_relpath"], shared_root / second_manifest["shared"]["link_relpath"]]
    assert all(path.is_symlink() for path in link_paths)
    assert len({path.name for path in link_paths}) == 2

    dashboard_html = shared_root / "registry" / "dashboard.html"
    dashboard_json = shared_root / "registry" / "dashboard_data.json"
    assert dashboard_html.exists()
    payload = json.loads(dashboard_json.read_text(encoding="utf-8"))
    assert payload["summary"]["entries"] == 2
    assert payload["summary"]["contributors"] == 1
    assert {entry["status"] for entry in payload["entries"]} == {"active"}
    assert any(entry["artifact_links"].get("scores_csv") for entry in payload["entries"])


def test_commit_skips_duplicate_realpaths(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    shared_root = tmp_path / "shared"

    first = REGISTRY.main(
        [
            "--shared-root",
            str(shared_root),
            "commit",
            "--name",
            "Alice Example",
            "--notes",
            "first commit",
            str(checkpoint["checkpoint_dir"]),
        ]
    )
    second = REGISTRY.main(
        [
            "--shared-root",
            str(shared_root),
            "commit",
            "--name",
            "Alice Example",
            "--notes",
            "duplicate commit",
            str(checkpoint["checkpoint_dir"]),
        ]
    )

    assert first == 0
    assert second == 0
    assert len(list((shared_root / "registry" / "entries").glob("*.json"))) == 1
    assert len(list((shared_root / "registry" / "batches").glob("*.json"))) == 1


def test_commit_allows_subtask_override_for_nonstandard_config_paths(tmp_path):
    checkpoint = _make_checkpoint(
        tmp_path,
        task="custom_task",
        subtask="unused_subtask",
        inferable_paths=False,
    )
    shared_root = tmp_path / "shared"

    result = REGISTRY.main(
        [
            "--shared-root",
            str(shared_root),
            "commit",
            "--name",
            "Bob Example",
            "--notes",
            "override subtask",
            "--subtask",
            "manual_subtask",
            str(checkpoint["checkpoint_dir"]),
        ]
    )

    assert result == 0
    [entry_manifest] = list((shared_root / "registry" / "entries").glob("*.json"))
    manifest = json.loads(entry_manifest.read_text(encoding="utf-8"))
    assert manifest["subtask"] == "manual_subtask"


def test_preview_marks_broken_symlinks(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    shared_root = tmp_path / "shared"

    commit_result = REGISTRY.main(
        [
            "--shared-root",
            str(shared_root),
            "commit",
            "--name",
            "Carol Example",
            "--notes",
            "checkpoint to break later",
            str(checkpoint["checkpoint_dir"]),
        ]
    )
    assert commit_result == 0

    shutil.rmtree(checkpoint["checkpoint_dir"])
    preview_result = REGISTRY.main(["--shared-root", str(shared_root), "preview"])
    assert preview_result == 0

    payload = json.loads((shared_root / "registry" / "dashboard_data.json").read_text(encoding="utf-8"))
    assert payload["summary"]["broken_entries"] == 1
    assert payload["entries"][0]["status"] == "broken"
