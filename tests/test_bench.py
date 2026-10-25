import dataclasses
import json

from clausewise.bench import PRF, check_thresholds, results_table, run_benchmark


def test_prf_math():
    p = PRF()
    p.add({1, 2, 3}, {2, 3, 4})
    d = p.as_dict()
    assert (d["tp"], d["fp"], d["fn"]) == (2, 1, 1)
    assert d["precision"] == d["recall"] == d["f1"] == 0.6667
    assert PRF().as_dict()["f1"] == 0.0


def test_check_thresholds():
    results = {"a": {"b": 0.9}, "c": 0.5}
    assert check_thresholds(results, {"a.b": 0.8, "c": 0.5}) == []
    failures = check_thresholds(results, {"a.b": 0.95, "missing.key": 0.1})
    assert len(failures) == 2 and failures[0].startswith("a.b")


def test_benchmark_runs_and_is_deterministic(settings, tmp_path):
    small = dataclasses.replace(settings, corpus_size=40, data_dir=tmp_path / "data")
    first = run_benchmark(small, tmp_path / "out1")
    second = run_benchmark(small, tmp_path / "out2")
    for key in (
        "segmentation",
        "classification",
        "entities",
        "risk",
        "redlines",
        "retrieval",
        "qa",
        "grounding_checker",
        "contract_type",
    ):
        assert key in first
        assert first[key] == second[key], key
    assert first["corpus"]["sha256"] == second["corpus"]["sha256"]
    written = json.loads((tmp_path / "out1" / "results.json").read_text())
    assert written["corpus"]["n_contracts"] == 40 and written["corpus"]["n_test"] == 12
    table = (tmp_path / "out1" / "results.md").read_text()
    assert "Segmentation boundary F1" in table and table == results_table(first)
    assert first["qa"]["hybrid"]["citation_validity"] == 1.0
