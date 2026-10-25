"""Benchmark harness: build the corpus, evaluate every stage on the held-out split, write results.

Run with ``uv run python -m clausewise.bench``. Every number in README.md comes from the JSON this module
writes; ``--check`` compares the fresh results against ``benchmarks/thresholds.json`` and exits non-zero
when a metric regresses, which is how CI uses it as a quality gate. All randomness is seeded, so two runs
with the same seed and the same embedding backend produce byte-identical metric values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score

from clausewise import __version__
from clausewise.classify import ClauseClassifier, rule_label, split_contracts
from clausewise.config import Settings, load_settings
from clausewise.corpus.generator import load_or_generate
from clausewise.corpus.questions import VALUE_ANSWER_TYPES
from clausewise.embeddings import Embedder, embedder_name, load_embedder
from clausewise.entities import extract_entities
from clausewise.grounding import support_score
from clausewise.pipeline import infer_contract_type, train_classifier
from clausewise.qa import Answerer
from clausewise.redline import suggest_redlines
from clausewise.retrieval import ClauseRetriever
from clausewise.risk import detect_risks
from clausewise.schemas import Clause, Contract
from clausewise.segment import body_text, segment
from clausewise.textutil import normalize_ws, sentences

RESULT_FILES = ("results.json", "results.md")


@dataclass
class PRF:
    """Precision / recall / F1 accumulator."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, found: set[Any], gold: set[Any]) -> None:
        """Count one document's predicted and gold sets."""
        self.tp += len(found & gold)
        self.fp += len(found - gold)
        self.fn += len(gold - found)

    def as_dict(self) -> dict[str, float | int]:
        """Rounded precision, recall and F1 plus the raw counts."""
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class Timer:
    """Wall-clock seconds per stage."""

    stages: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    def lap(self, name: str) -> None:
        """Record the time since the previous lap under ``name``."""
        now = time.perf_counter()
        self.stages[name] = round(now - self._start, 3)
        self._start = now


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset)


def _is_noisy(c: Contract) -> bool:
    return float(c.style.get("heading_dropout", 0)) > 0


def eval_segmentation(test: list[Contract]) -> dict[str, Any]:
    """Boundary precision/recall/F1 at line resolution, plus exact span match rate, split clean/noisy."""
    by_cond = {"all": PRF(), "clean": PRF(), "noisy": PRF()}
    exact = total = 0
    for c in test:
        pred = segment(c.text)
        gold = {_line_of(c.text, cl.start) for cl in c.clauses}
        found = {_line_of(c.text, cl.start) for cl in pred}
        by_cond["all"].add(found, gold)
        by_cond["noisy" if _is_noisy(c) else "clean"].add(found, gold)
        gold_spans = {(cl.start, cl.end) for cl in c.clauses}
        exact += sum(1 for cl in pred if (cl.start, cl.end) in gold_spans)
        total += len(c.clauses)
    out: dict[str, Any] = {cond: prf.as_dict() for cond, prf in by_cond.items()}
    out["exact_span_rate"] = round(exact / total, 4) if total else 0.0
    out["n_gold_clauses"] = total
    return out


def _strip_headings(docs: list[list[Clause]]) -> list[list[Clause]]:
    return [[cl.model_copy(update={"heading": None}) for cl in doc] for doc in docs]


def eval_classification(
    train: list[Contract], test: list[Contract], settings: Settings
) -> tuple[dict[str, Any], ClauseClassifier]:
    """Macro-F1 of the trained model (with and without headings) and of the heading-rule baseline."""
    clf = train_classifier(train + test, settings, use_heading=True)
    test_docs = [c.clauses for c in test]
    y_true = [cl.label or "" for doc in test_docs for cl in doc]
    y_full = clf.predict_docs(test_docs)
    clf_body = train_classifier(train + test, settings, use_heading=False)
    y_body = clf_body.predict_docs(_strip_headings(test_docs))
    y_rule = [rule_label(cl, i, len(doc)) for doc in test_docs for i, cl in enumerate(doc)]
    labels = sorted(set(y_true))
    per_class = f1_score(y_true, y_body, labels=labels, average=None, zero_division=0)
    return {
        "n_train_clauses": sum(len(c.clauses) for c in train),
        "n_test_clauses": len(y_true),
        "n_labels": len(labels),
        "macro_f1_heading_and_body": round(float(f1_score(y_true, y_full, average="macro", zero_division=0)), 4),
        "accuracy_heading_and_body": round(sum(a == b for a, b in zip(y_true, y_full, strict=True)) / len(y_true), 4),
        "macro_f1_body_only": round(float(f1_score(y_true, y_body, average="macro", zero_division=0)), 4),
        "accuracy_body_only": round(sum(a == b for a, b in zip(y_true, y_body, strict=True)) / len(y_true), 4),
        "macro_f1_rule_baseline": round(float(f1_score(y_true, y_rule, average="macro", zero_division=0)), 4),
        "weakest_classes_body_only": sorted(
            zip(labels, [round(float(x), 4) for x in per_class], strict=True), key=lambda x: x[1]
        )[:5],
    }, clf


def eval_entities(test: list[Contract]) -> dict[str, Any]:
    """Mention-level P/R/F1 per entity type: same type, same normalised value, overlapping span."""
    per_type: dict[str, PRF] = defaultdict(PRF)
    overall = PRF()
    for c in test:
        pred = extract_entities(c.text)
        gold = list(c.entities)
        matched: set[int] = set()
        for p in pred:
            hit = next(
                (
                    i
                    for i, g in enumerate(gold)
                    if i not in matched
                    and g.type == p.type
                    and g.value == p.value
                    and g.start < p.end
                    and p.start < g.end
                ),
                None,
            )
            if hit is None:
                per_type[p.type].fp += 1
                overall.fp += 1
            else:
                matched.add(hit)
                per_type[p.type].tp += 1
                overall.tp += 1
        for i, g in enumerate(gold):
            if i not in matched:
                per_type[g.type].fn += 1
                overall.fn += 1
    out: dict[str, Any] = {t: prf.as_dict() for t, prf in sorted(per_type.items())}
    out["all"] = overall.as_dict()
    return out


def _risk_set(findings: list[Any]) -> set[tuple[str, str]]:
    return {(f.risk_type, f.clause_id) for f in findings}


def _overlapping_gold(c: Contract, span: tuple[int, int]) -> str | None:
    return next(
        (
            g.clause_id
            for g in c.clauses
            if g.start < span[1]
            and span[0] < g.end
            and (min(g.end, span[1]) - max(g.start, span[0])) > 0.5 * (g.end - g.start)
        ),
        None,
    )


def eval_risks(test: list[Contract]) -> dict[str, Any]:
    """Risk P/R/F1 on gold clause spans and end to end from raw text, overall and per risk type."""
    gold_seg = PRF()
    e2e = {"all": PRF(), "clean": PRF(), "noisy": PRF()}
    per_type: dict[str, PRF] = defaultdict(PRF)
    for c in test:
        gold = {(r.risk_type, r.clause_id) for r in c.risks}
        found = _risk_set(detect_risks(c.clauses))
        gold_seg.add(found, gold)
        for rt in {x[0] for x in gold | found}:
            per_type[rt].add({x for x in found if x[0] == rt}, {x for x in gold if x[0] == rt})
        pred_clauses = segment(c.text)
        mapped = {
            (f.risk_type, _overlapping_gold(c, (cl.start, cl.end)))
            for f in detect_risks(pred_clauses)
            for cl in pred_clauses
            if cl.clause_id == f.clause_id
        }
        e2e["all"].add(mapped, gold)
        e2e["noisy" if _is_noisy(c) else "clean"].add(mapped, gold)
    return {
        "gold_segments": gold_seg.as_dict(),
        "per_type_gold_segments": {t: prf.as_dict() for t, prf in sorted(per_type.items())},
        "end_to_end": {cond: prf.as_dict() for cond, prf in e2e.items()},
        "n_planted": sum(len(c.risks) for c in test),
    }


def eval_redlines(test: list[Contract]) -> dict[str, Any]:
    """Share of findings with a redline, and share of redlines whose proposed text clears the detector."""
    total = resolved = 0
    for c in test:
        findings = detect_risks(c.clauses)
        redlines = suggest_redlines(c.clauses, findings, (c.parties[0].role, c.parties[1].role))
        for r in redlines:
            total += 1
            again = detect_risks(
                [Clause(clause_id=r.clause_id, start=0, end=len(r.proposed), heading=None, text=r.proposed)]
            )
            resolved += not any(f.risk_type == r.risk_type for f in again)
    return {"n_redlines": total, "resolved_rate": round(resolved / total, 4) if total else 0.0}


def eval_retrieval(test: list[Contract], embedder: Embedder | None) -> dict[str, Any]:
    """Recall@1, Recall@5 and MRR of the gold clause for answerable questions, per backend."""
    out: dict[str, Any] = {}
    configs: list[tuple[str, str, Embedder | None]] = [
        ("bm25", "bm25", None),
        ("tfidf", "tfidf", None),
        ("hybrid_bm25_tfidf", "hybrid", None),
    ]
    if embedder is not None:
        configs += [("dense", "dense", embedder), ("hybrid_bm25_dense", "hybrid", embedder)]
    for name, mode, emb in configs:
        r1 = r5 = n = 0
        mrr = 0.0
        backend = ""
        for c in test:
            retriever = ClauseRetriever(c.clauses, emb, mode)
            backend = retriever.backend
            for q in c.questions:
                if not q.answerable:
                    continue
                ranked = [c.clauses[i].clause_id for i in retriever.rank(q.question)]
                n += 1
                r1 += ranked[:1] == [q.gold_clause_id]
                r5 += q.gold_clause_id in ranked[:5]
                mrr += 1.0 / (ranked.index(q.gold_clause_id) + 1)
        out[name] = {
            "backend": backend,
            "recall_at_1": round(r1 / n, 4),
            "recall_at_5": round(r5 / n, 4),
            "mrr": round(mrr / n, 4),
            "n_questions": n,
        }
    out["dense_backend"] = embedder_name(embedder)
    return out


def eval_qa(
    test: list[Contract], embedder: Embedder | None, settings: Settings, mode: str = "hybrid"
) -> dict[str, Any]:
    """Citation validity, citation accuracy, exact match on value questions, abstention behaviour."""
    stats: Counter[str] = Counter()
    em: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    backend = ""
    for c in test:
        retriever = ClauseRetriever(c.clauses, embedder, mode)
        backend = retriever.backend
        answerer = Answerer(c.clauses, retriever, settings)
        valid_ids = {cl.clause_id for cl in c.clauses}
        for q in c.questions:
            ans = answerer.answer(q.question)
            if not q.answerable:
                stats["unanswerable"] += 1
                stats["abstained_on_unanswerable"] += ans.abstained
                continue
            stats["answerable"] += 1
            if ans.abstained:
                stats["abstained_on_answerable"] += 1
                continue
            stats["answered"] += 1
            cited = {cid for s in ans.sentences for cid in s.citations}
            stats["citations_valid"] += (
                bool(ans.sentences) and cited <= valid_ids and all(s.grounded for s in ans.sentences)
            )
            stats["cites_gold_clause"] += q.gold_clause_id in cited
            if q.answer_type in VALUE_ANSWER_TYPES and q.gold_value is not None:
                em[q.answer_type][1] += 1
                em[q.answer_type][0] += ans.short_answer == q.gold_value
    answered = max(1, stats["answered"])
    em_total = sum(v[1] for v in em.values())
    return {
        "backend": backend,
        "n_questions": stats["answerable"] + stats["unanswerable"],
        "answered_rate": round(stats["answered"] / stats["answerable"], 4),
        "citation_validity": round(stats["citations_valid"] / answered, 4),
        "citation_accuracy": round(stats["cites_gold_clause"] / answered, 4),
        "exact_match_value_questions": round(sum(v[0] for v in em.values()) / em_total, 4) if em_total else 0.0,
        "exact_match_by_type": {k: {"em": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(em.items())},
        "abstention_on_unanswerable": round(stats["abstained_on_unanswerable"] / max(1, stats["unanswerable"]), 4),
        "n_unanswerable": stats["unanswerable"],
    }


def _perturb(rng: random.Random, sentence: str, other: str) -> list[tuple[str, str]]:
    """Negative examples for the grounding checker: each tuple is (kind, perturbed sentence)."""
    out: list[tuple[str, str]] = [("other_clause", other)]
    nums = [m for m in __import__("re").finditer(r"\d+", sentence)]
    if nums:
        m = rng.choice(nums)
        new = str(int(m.group(0)) + rng.choice((1, 5, 15, 30)))
        out.append(("number_swap", sentence[: m.start()] + new + sentence[m.end() :]))
    words = sentence.split()
    if len(words) > 4:
        idx = next((i for i, w in enumerate(words) if w.lower() in {"shall", "may", "will", "is", "are"}), None)
        if idx is not None and idx + 1 < len(words):
            if words[idx + 1].lower() == "not":
                out.append(("negation_removed", " ".join(words[: idx + 1] + words[idx + 2 :])))
            else:
                out.append(("negation_inserted", " ".join([*words[: idx + 1], "not", *words[idx + 1 :]])))
    out.append(("fabricated", "The Supplier shall provide all upgrades free of charge for the life of the Agreement."))
    return out


def eval_grounding(test: list[Contract], settings: Settings) -> dict[str, Any]:
    """Accept/reject accuracy of the checker on verbatim sentences and on controlled perturbations."""
    rng = random.Random(settings.seed)
    accepted_pos = n_pos = 0
    rejected: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for c in test:
        bodies = [(cl, body_text(cl)) for cl in c.clauses if cl.label not in {"preamble", "signature"}]
        for cl, body in bodies:
            for s in sentences(body)[:2]:
                sentence = normalize_ws(s)
                if len(sentence.split()) < 5:
                    continue
                n_pos += 1
                variant = sentence.upper() if rng.random() < 0.3 else sentence.replace(" ", "\n", 1)
                accepted_pos += support_score(variant, cl.text, settings.grounding_threshold).supported
                other_cl = rng.choice([x for x in bodies if x[0] is not cl])
                other = normalize_ws(sentences(other_cl[1])[0])
                for kind, neg in _perturb(rng, sentence, other):
                    totals[kind] += 1
                    rejected[kind] += not support_score(neg, cl.text, settings.grounding_threshold).supported
    n_neg = sum(totals.values())
    return {
        "n_positive": n_pos,
        "accept_rate_verbatim": round(accepted_pos / n_pos, 4),
        "n_negative": n_neg,
        "reject_rate_all_perturbations": round(sum(rejected.values()) / n_neg, 4),
        "reject_rate_by_kind": {k: round(rejected[k] / totals[k], 4) for k in sorted(totals)},
    }


def eval_contract_type(test: list[Contract]) -> dict[str, Any]:
    """Accuracy of contract-type inference from raw text and gold labels of segmented clauses."""
    correct = sum(infer_contract_type(c.text, [cl.label or "" for cl in c.clauses]) == c.contract_type for c in test)
    return {"accuracy": round(correct / len(test), 4), "n": len(test)}


def corpus_digest(contracts: list[Contract]) -> str:
    """SHA-256 over all contract texts, the determinism fingerprint of the corpus."""
    h = hashlib.sha256()
    for c in contracts:
        h.update(c.text.encode("utf-8"))
    return h.hexdigest()


def run_benchmark(settings: Settings, out_dir: Path) -> dict[str, Any]:
    """Run every evaluation and write results.json and results.md into ``out_dir``."""
    timer = Timer()
    contracts = load_or_generate(settings.data_dir, settings.seed, settings.corpus_size)
    timer.lap("corpus")
    train, test = split_contracts(contracts, settings.seed)
    embedder = load_embedder(settings)
    results: dict[str, Any] = {
        "version": __version__,
        "seed": settings.seed,
        "corpus": {
            "n_contracts": len(contracts),
            "n_train": len(train),
            "n_test": len(test),
            "n_clauses": sum(len(c.clauses) for c in contracts),
            "sha256": corpus_digest(contracts),
            "types": dict(sorted(Counter(c.contract_type for c in contracts).items())),
            "planted_risks": dict(sorted(Counter(r.risk_type for c in contracts for r in c.risks).items())),
            "n_noisy_test": sum(_is_noisy(c) for c in test),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embeddings": embedder_name(embedder),
        },
    }
    results["segmentation"] = eval_segmentation(test)
    timer.lap("segmentation")
    results["classification"], _clf = eval_classification(train, test, settings)
    timer.lap("classification")
    results["entities"] = eval_entities(test)
    timer.lap("entities")
    results["risk"] = eval_risks(test)
    timer.lap("risk")
    results["redlines"] = eval_redlines(test)
    timer.lap("redlines")
    results["contract_type"] = eval_contract_type(test)
    results["retrieval"] = eval_retrieval(test, embedder)
    timer.lap("retrieval")
    results["qa"] = {
        "hybrid": eval_qa(test, embedder, settings, "hybrid"),
        "bm25": eval_qa(test, None, settings, "bm25"),
    }
    timer.lap("qa")
    results["grounding_checker"] = eval_grounding(test, settings)
    timer.lap("grounding")
    results["timings_seconds"] = timer.stages
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(results_table(results), encoding="utf-8")
    return results


def results_table(r: dict[str, Any]) -> str:
    """The Markdown table that README.md embeds verbatim."""
    seg, cls, ent, risk, ret, qa, gc = (
        r[k] for k in ("segmentation", "classification", "entities", "risk", "retrieval", "qa", "grounding_checker")
    )
    dense = ret.get("hybrid_bm25_dense")
    rows = [
        (
            "Segmentation boundary F1 (all / clean / noisy)",
            f"{seg['all']['f1']:.4f} / {seg['clean']['f1']:.4f} / {seg['noisy']['f1']:.4f}",
        ),
        ("Segmentation exact span rate", f"{seg['exact_span_rate']:.4f}"),
        ("Classification macro-F1, heading + body", f"{cls['macro_f1_heading_and_body']:.4f}"),
        ("Classification macro-F1, body only", f"{cls['macro_f1_body_only']:.4f}"),
        ("Classification macro-F1, heading-rule baseline", f"{cls['macro_f1_rule_baseline']:.4f}"),
        ("Entity extraction F1 (all types)", f"{ent['all']['f1']:.4f}"),
        (
            "Risk detection precision / recall (gold segments)",
            f"{risk['gold_segments']['precision']:.4f} / {risk['gold_segments']['recall']:.4f}",
        ),
        (
            "Risk detection precision / recall (end to end)",
            f"{risk['end_to_end']['all']['precision']:.4f} / {risk['end_to_end']['all']['recall']:.4f}",
        ),
        ("Redlines that clear the detector", f"{r['redlines']['resolved_rate']:.4f}"),
        ("Retrieval Recall@5: BM25 / TF-IDF", f"{ret['bm25']['recall_at_5']:.4f} / {ret['tfidf']['recall_at_5']:.4f}"),
        (
            "Retrieval Recall@5: dense / hybrid BM25+dense",
            f"{ret['dense']['recall_at_5']:.4f} / {dense['recall_at_5']:.4f}" if dense else "not run (no embedder)",
        ),
        ("QA citation validity (hybrid)", f"{qa['hybrid']['citation_validity']:.4f}"),
        ("QA citation accuracy, gold clause cited (hybrid)", f"{qa['hybrid']['citation_accuracy']:.4f}"),
        ("QA exact match on value questions (hybrid)", f"{qa['hybrid']['exact_match_value_questions']:.4f}"),
        ("QA abstention on unanswerable questions (hybrid)", f"{qa['hybrid']['abstention_on_unanswerable']:.4f}"),
        (
            "Grounding checker: verbatim accepted / perturbations rejected",
            f"{gc['accept_rate_verbatim']:.4f} / {gc['reject_rate_all_perturbations']:.4f}",
        ),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"| {'Metric':<{width}} | Value |", f"| {'-' * width} | ----- |"]
    lines += [f"| {name:<{width}} | {value} |" for name, value in rows]
    n_test, n_questions = r["corpus"]["n_test"], r["qa"]["hybrid"]["n_questions"]
    header = (
        f"Measured on {n_test} held-out contracts ({n_questions} questions), seed {r['seed']}, "
        f"embeddings: {r['environment']['embeddings']}.\n\n"
    )
    return header + "\n".join(lines) + "\n"


def _lookup(results: dict[str, Any], dotted: str) -> float:
    node: Any = results
    for part in dotted.split("."):
        node = node[part]
    return float(node)


def check_thresholds(results: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Return a failure message per metric below its threshold."""
    failures = []
    for key, minimum in thresholds.items():
        try:
            value = _lookup(results, key)
        except (KeyError, TypeError):
            failures.append(f"{key}: missing from results")
            continue
        if value < minimum:
            failures.append(f"{key}: {value:.4f} < {minimum:.4f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the clausewise benchmark")
    parser.add_argument("--out", type=Path, default=Path("benchmarks"))
    parser.add_argument("--check", type=Path, default=None, help="thresholds JSON; exit 1 if any metric is below")
    args = parser.parse_args(argv)
    settings = load_settings()
    results = run_benchmark(settings, args.out)
    print(results_table(results))
    print("timings (s):", json.dumps(results["timings_seconds"]))
    if args.check:
        failures = check_thresholds(results, json.loads(args.check.read_text(encoding="utf-8")))
        if failures:
            print("THRESHOLD FAILURES:\n  " + "\n  ".join(failures))
            return 1
        print(f"all {len(json.loads(args.check.read_text(encoding='utf-8')))} thresholds met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
