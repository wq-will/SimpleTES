"""Test per-chain construction tracking logic."""
import os
import json
import tempfile


def test_per_chain_best_scores_initialization():
    """Verify _chain_best_scores dict is initialized correctly."""
    # Simulate what the engine does at init
    num_chains = 4
    chain_best_scores = {i: -float("inf") for i in range(max(1, num_chains))}

    assert len(chain_best_scores) == 4
    assert all(score == -float("inf") for score in chain_best_scores.values())
    print("  ✓ Initialization")


def test_per_chain_update_logic():
    """Verify per-chain update only affects the target chain."""
    chain_best_scores = {0: 0.5, 1: 0.3, 2: -float("inf")}
    chain_constructions = {0: None, 1: None, 2: None}

    # Simulate a node in chain 1 beating chain 1's best
    node_chain_idx = 1
    node_score = 0.6
    if node_chain_idx in chain_best_scores and node_score > chain_best_scores[node_chain_idx]:
        chain_best_scores[node_chain_idx] = node_score
        chain_constructions[node_chain_idx] = {"snapshot_id": "n1", "path": "/tmp/n1.json"}

    assert chain_best_scores[0] == 0.5, "Chain 0 should be unchanged"
    assert chain_best_scores[1] == 0.6, "Chain 1 should be updated"
    assert chain_best_scores[2] == -float("inf"), "Chain 2 should be unchanged"
    assert chain_constructions[0] is None, "Chain 0 construction should be unchanged"
    assert chain_constructions[1] is not None, "Chain 1 construction should be set"
    assert chain_constructions[2] is None, "Chain 2 construction should be unchanged"
    print("  ✓ Per-chain update isolation")


def test_per_chain_no_update_when_not_improving():
    """Verify no update when node doesn't beat chain best."""
    chain_best_scores = {0: 0.8, 1: 0.5}
    chain_constructions = {0: {"snapshot_id": "old"}, 1: None}

    # Node in chain 0 with lower score
    node_chain_idx = 0
    node_score = 0.6
    if node_chain_idx in chain_best_scores and node_score > chain_best_scores[node_chain_idx]:
        chain_best_scores[node_chain_idx] = node_score
        chain_constructions[node_chain_idx] = {"snapshot_id": "new"}

    assert chain_best_scores[0] == 0.8, "Chain 0 best should not decrease"
    assert chain_constructions[0]["snapshot_id"] == "old", "Chain 0 construction should not change"
    print("  ✓ No update when not improving")


def test_checkpoint_serialization():
    """Verify chain_best_scores round-trips through checkpoint format."""
    original = {0: 0.5, 1: 0.7, 2: -float("inf")}

    # Serialize (what _write_checkpoint does)
    serialized = {str(k): v for k, v in original.items()}
    json_str = json.dumps(serialized)

    # Deserialize (what load_checkpoint does)
    restored_raw = json.loads(json_str)
    num_chains = 3
    restored = {i: -float("inf") for i in range(num_chains)}
    for chain_idx_str, score in restored_raw.items():
        try:
            chain_idx = int(chain_idx_str)
        except (TypeError, ValueError):
            continue
        if 0 <= chain_idx < num_chains:
            restored[chain_idx] = float(score)

    assert restored == original
    print("  ✓ Checkpoint serialization round-trip")


def test_backward_compat_no_chain_best_scores():
    """Verify backward compat when checkpoint lacks chain_best_scores."""
    # Simulate old checkpoint with no chain_best_scores
    restored_chain_best = {}
    num_chains = 3
    chain_best_scores = {i: -float("inf") for i in range(num_chains)}

    # Simulate policy chain data for backward compat
    # chains maps chain_idx -> list of node_ids
    node_scores = {"n1": 0.3, "n2": 0.5, "n3": 0.8, "n4": 0.4}
    chains = {"0": ["n1", "n3"], "1": ["n2", "n4"]}

    if not restored_chain_best:
        for chain_idx_str, node_ids in chains.items():
            try:
                chain_idx = int(chain_idx_str)
            except (TypeError, ValueError):
                continue
            if 0 <= chain_idx < len(chain_best_scores):
                best_in_chain = max(
                    (node_scores[nid] for nid in node_ids if nid in node_scores),
                    default=-float("inf"),
                )
                chain_best_scores[chain_idx] = best_in_chain

    assert chain_best_scores[0] == 0.8, "Chain 0 best should be max of n1, n3"
    assert chain_best_scores[1] == 0.5, "Chain 1 best should be max of n2, n4"
    assert chain_best_scores[2] == -float("inf"), "Chain 2 should remain uninitialized"
    print("  ✓ Backward compatibility reconstruction")


def test_os_path_exists_guard():
    """Verify os.path.exists guard prevents redundant writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "snapshot.json")

        # First write
        write_count = 0
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"data": "first"}, f)
            write_count += 1

        # Second call with same path (simulates another chain referencing same snapshot)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"data": "second"}, f)
            write_count += 1

        assert write_count == 1, "Should only write once"
        with open(path) as f:
            assert json.load(f)["data"] == "first"
        print("  ✓ os.path.exists guard prevents redundant writes")


if __name__ == "__main__":
    print("Per-chain construction tracking tests:")
    test_per_chain_best_scores_initialization()
    test_per_chain_update_logic()
    test_per_chain_no_update_when_not_improving()
    test_checkpoint_serialization()
    test_backward_compat_no_chain_best_scores()
    test_os_path_exists_guard()
    print("\nAll tests passed!")
