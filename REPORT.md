# REPORT

What was built, how every number was measured, and what is still missing. All commands were run from
a local checkout on macOS (Apple Silicon, CPU only) with Python 3.13.17 via uv,
seed 42, on 2026-10-25.

## What was built

A contract-intelligence pipeline, package `clausewise`, with:

- a seeded corpus generator (400 contracts, 6809 clauses, 5 contract types, 37 clause
  labels, 6 planted risk types with hard negatives, exact entity spans, 3600 paraphrased questions
  of which 400 are unanswerable, half of the contracts with dropped headings and letter-swap noise);
- clause segmentation, a TF-IDF plus structural-cue logistic-regression clause classifier with a
  heading-rule baseline, regex entity extraction with normalisation, a sentence-scoped risk rule engine
  driven by a data-only playbook, redline suggestions with diffs;
- retrieval over clauses (BM25, TF-IDF, model2vec dense embeddings, reciprocal-rank-fusion hybrid),
  an extractive answerer that only returns verbatim sentences with clause ids, a grounding checker
  (verbatim or ordered-token match, exact numbers, negation counts in the aligned span), and an optional
  Anthropic answerer behind `CLAUSEWISE_LLM=1` that passes through the same checker;
- a benchmark harness that writes `benchmarks/results.json` and `results.md` and acts as a CI gate
  against `benchmarks/thresholds.json`;
- a FastAPI service with typed schemas, request ids, JSON logs, Prometheus metrics, API-key auth and a
  vanilla-JS dashboard; a CLI; a multi-stage non-root Dockerfile with the embedding model baked in;
  GitHub Actions CI; 215 tests.

## Measured results

Command: `make bench` (equivalent to `uv run python -m clausewise.bench --out benchmarks`, then
`scripts/check_readme_numbers.py`). Embeddings backend: `model2vec:minishlab/potion-base-8M`. Test split:
120 contracts (62 noisy), 2000 clauses, 1080 questions. Corpus digest `bddd71c615f3e41d...`.

Output tail of `make bench`:

```
uv run python -m clausewise.bench --out benchmarks
Measured on 120 held-out contracts (1080 questions), seed 42, embeddings: model2vec:minishlab/potion-base-8M.

| Metric                                                        | Value |
| ------------------------------------------------------------- | ----- |
| Segmentation boundary F1 (all / clean / noisy)                | 0.9977 / 1.0000 / 0.9956 |
| Segmentation exact span rate                                  | 0.9910 |
| Classification macro-F1, heading + body                       | 1.0000 |
| Classification macro-F1, body only                            | 1.0000 |
| Classification macro-F1, heading-rule baseline                | 0.9454 |
| Entity extraction F1 (all types)                              | 0.9995 |
| Risk detection precision / recall (gold segments)             | 1.0000 / 1.0000 |
| Risk detection precision / recall (end to end)                | 1.0000 / 1.0000 |
| Redlines that clear the detector                              | 1.0000 |
| Retrieval Recall@5: BM25 / TF-IDF                             | 0.8177 / 0.8177 |
| Retrieval Recall@5: dense / hybrid BM25+dense                 | 0.8833 / 0.9146 |
| QA citation validity (hybrid)                                 | 1.0000 |
| QA citation accuracy, gold clause cited (hybrid)              | 0.9052 |
| QA exact match on value questions (hybrid)                    | 0.9378 |
| QA abstention on unanswerable questions (hybrid)              | 0.5833 |
| Grounding checker: verbatim accepted / perturbations rejected | 1.0000 / 0.9811 |

timings (s): {"corpus": 0.032, "segmentation": 0.175, "classification": 2.25, "entities": 0.108, "risk": 0.586, "redlines": 0.325, "retrieval": 0.917, "qa": 51.774, "grounding": 0.77}
uv run python scripts/check_readme_numbers.py
README results table matches benchmarks/results.md
```

### Detail

Segmentation: 1991 boundaries recovered, 0 false boundaries, 9 missed (all 9 misses are in noisy
contracts, where a clause lost its heading and followed a sub-numbered clause). Exact span rate 0.9910
over 2000 gold clauses.

Classification (4809 training clauses, 2000 test clauses, 37 labels): macro-F1 1.0000 with
headings, 1.0000 body only, accuracy 1.0000 body only; heading-rule baseline 0.9454.

Entities (mention level, same type and normalised value, overlapping span):

| type | precision | recall | F1 | tp | fp | fn |
| --- | --- | --- | --- | --- | --- | --- |
| date | 1.0000 | 1.0000 | 1.0000 | 120 | 0 | 0 |
| duration | 1.0000 | 1.0000 | 1.0000 | 871 | 0 | 0 |
| jurisdiction | 1.0000 | 1.0000 | 1.0000 | 120 | 0 | 0 |
| money | 1.0000 | 1.0000 | 1.0000 | 155 | 0 | 0 |
| party | 1.0000 | 0.9958 | 0.9979 | 478 | 0 | 2 |
| percentage | 1.0000 | 1.0000 | 1.0000 | 95 | 0 | 0 |
| all | 1.0000 | 0.9989 | 0.9995 | 1839 | 0 | 2 |

Risk detection on gold segments (195 planted risks in the test split):

| risk type | tp | fp | fn |
| --- | --- | --- | --- |
| auto_renewal | 36 | 0 | 0 |
| governing_law | 55 | 0 | 0 |
| ip_assignment | 32 | 0 | 0 |
| non_compete | 19 | 0 | 0 |
| unilateral_termination | 32 | 0 | 0 |
| unlimited_liability | 21 | 0 | 0 |

End to end from raw text: precision 1.0000, recall 1.0000 (clean 1.0000, noisy 1.0000).
Redlines: 195 generated, 1.0000 clear the detector. Contract type accuracy 1.0000.

Retrieval (960 answerable questions):

| config | backend | Recall@1 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| bm25 | bm25 | 0.6083 | 0.8177 | 0.7099 |
| tfidf | tfidf | 0.6146 | 0.8177 | 0.7130 |
| hybrid_bm25_tfidf | bm25+tfidf | 0.6083 | 0.8177 | 0.7097 |
| dense | dense | 0.6813 | 0.8833 | 0.7730 |
| hybrid_bm25_dense | bm25+dense | 0.6615 | 0.9146 | 0.7689 |

QA (1080 questions, 120 unanswerable):

| metric | hybrid BM25+dense | BM25 only | TF-IDF fallback run |
| --- | --- | --- | --- |
| answered_rate | 0.9563 | 0.9510 | 0.9510 |
| citation_validity | 1.0000 | 1.0000 | 1.0000 |
| citation_accuracy | 0.9052 | 0.8817 | 0.8806 |
| exact_match_value_questions | 0.9378 | 0.9257 | 0.9257 |
| abstention_on_unanswerable | 0.5833 | 0.5833 | 0.5833 |

Exact match by answer type (hybrid):

| type | EM | n |
| --- | --- | --- |
| boolean | 0.9712 | 104 |
| date | 1.0000 | 65 |
| duration | 0.8632 | 212 |
| jurisdiction | 1.0000 | 64 |
| money | 1.0000 | 58 |
| money_or_fees | 0.6667 | 15 |
| parties | 1.0000 | 71 |
| percentage | 1.0000 | 6 |

Grounding checker (3271 verbatim positives accepted at 1.0000; 10039 perturbations):

| perturbation | reject rate |
| --- | --- |
| fabricated | 1.0000 |
| negation_inserted | 0.9447 |
| negation_removed | 0.8774 |
| number_swap | 0.9674 |
| other_clause | 0.9994 |

### TF-IDF fallback run

Command: `CLAUSEWISE_EMBEDDINGS=tfidf uv run python -m clausewise.bench --out benchmarks/tmp/tfidf`.
This is what CI measures if the embedding model cannot be downloaded. Backend-independent metrics are
identical to the dense run; QA citation accuracy drops to 0.8806 and exact match to 0.9257.

```
Measured on 120 held-out contracts (1080 questions), seed 42, embeddings: tfidf-fallback.

| Metric                                                        | Value |
| ------------------------------------------------------------- | ----- |
| Segmentation boundary F1 (all / clean / noisy)                | 0.9977 / 1.0000 / 0.9956 |
| Segmentation exact span rate                                  | 0.9910 |
| Classification macro-F1, heading + body                       | 1.0000 |
| Classification macro-F1, body only                            | 1.0000 |
| Classification macro-F1, heading-rule baseline                | 0.9454 |
| Entity extraction F1 (all types)                              | 0.9995 |
| Risk detection precision / recall (gold segments)             | 1.0000 / 1.0000 |
| Risk detection precision / recall (end to end)                | 1.0000 / 1.0000 |
| Redlines that clear the detector                              | 1.0000 |
| Retrieval Recall@5: BM25 / TF-IDF                             | 0.8177 / 0.8177 |
| Retrieval Recall@5: dense / hybrid BM25+dense                 | not run (no embedder) |
| QA citation validity (hybrid)                                 | 1.0000 |
| QA citation accuracy, gold clause cited (hybrid)              | 0.8806 |
| QA exact match on value questions (hybrid)                    | 0.9257 |
| QA abstention on unanswerable questions (hybrid)              | 0.5833 |
| Grounding checker: verbatim accepted / perturbations rejected | 1.0000 / 0.9811 |

timings (s): {"corpus": 0.04, "segmentation": 0.048, "classification": 2.964, "entities": 0.138, "risk": 0.768, "redlines": 0.424, "retrieval": 0.815, "qa": 50.621, "grounding": 0.637}
```

### Determinism

Two full dense runs (`benchmarks/results.json` and `benchmarks/tmp/run2/results.json`) were compared
after dropping `timings_seconds`: identical, including the corpus digest. The 40-contract version of the
same check runs in `tests/test_bench.py::test_benchmark_runs_and_is_deterministic`.

### CI gate

Command: `uv run python -m clausewise.bench --out benchmarks/tmp/check --check benchmarks/thresholds.json`

```
timings (s): {"corpus": 0.028, "segmentation": 0.216, "classification": 2.323, "entities": 0.107, "risk": 0.61, "redlines": 0.34, "retrieval": 0.896, "qa": 50.817, "grounding": 0.757}
all 18 thresholds met
```

## Lint and tests

`make lint` (ruff check, ruff format --check, `scripts/check_quality.py`: functions under 60 lines, no
em or en dashes, no placeholder markers):

```
uv run ruff check .
All checks passed!
uv run ruff format --check .
51 files already formatted
uv run python scripts/check_quality.py
check_quality: 0 problem(s)
lint exit=0
```

`make test`:

```
........................................................................ [ 33%]
........................................................................ [ 66%]
s......................................................................  [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_retrieval.py:93: set CLAUSEWISE_TEST_DENSE=1 to test the cached dense model
214 passed, 1 skipped in 29.23s
```

214 passed, 1 skipped (the dense-backend test runs only with `CLAUSEWISE_TEST_DENSE=1`). The suite
includes property-based tests (hypothesis) for the tokeniser, generator invariants, segmenter, entity
round trips, BM25 and the grounding checker; adversarial tests (prompt injection inside a contract,
binary garbage, oversized payloads, hard-negative risk sentences, fabricated and number-swapped LLM
output); and determinism tests (same seed, same corpus and same benchmark).

## Service checks

- `make serve` (`uv run clausewise serve`) was started on a free port; `/health`, `/api/samples`,
  `/api/analyze`, `/api/ask`, `/metrics`, `/docs` and `/` all returned 200. Request ids were echoed, log
  lines were JSON, `/health` reported `model2vec:minishlab/potion-base-8M` after first use.
- `docker build -t clausewise:latest .` succeeded (image 718 MB, python:3.13-slim runtime, user `app`
  uid 999, model cached under `/app/hf`). The container answered `/health` within about 4 seconds of
  start, analysed a short contract in 3.6 s including first-use classifier training (found
  auto_renewal, unlimited_liability, governing_law, score 55) and answered "Does the agreement renew
  automatically?" with `yes` citing `c02`.

## Known gaps and honest caveats

1. **Synthetic ceiling.** Classification macro-F1 of 1.0 and risk precision/recall of 1.0 on gold
   segments mean the rules and model cover every template variant, including the hard negatives. They
   are not evidence about real contracts, and the generator and the rules share an author. The
   rule baseline (0.9454), the noisy split, the retrieval and QA numbers, and the grounding
   perturbation suite are the informative measurements.
2. **Abstention on unanswerable questions is 0.5833.** The lexical answerer answers a rent question
   from a fee clause when both are monthly amounts. Reported as the weakest number; not tuned away.
3. **Duration exact match is 0.8632.** Neighbouring durations (cure period next to notice period,
   non-compete months next to term) are sometimes picked.
4. **Grounding checker limits.** It rejects 98.1% of perturbations overall but only 87.7% of removed
   negations and 94.5% of inserted ones; a verbatim sentence that answers a different question passes.
5. **LLM path is untested against the live API.** It is exercised with a fake client only, by design
   (no network in tests or benchmarks). Default model `claude-haiku-5-5` via `CLAUSEWISE_LLM_MODEL`.
6. **No real-data evaluation and no document parsing.** Plain text only; PDF and DOCX extraction, and a
   public labelled contract set, are the first two roadmap items.
7. **Docker image is 718 MB** because of scikit-learn, scipy and numpy on python:3.13-slim; no attempt
   was made to slim it further.
8. **Lint passes with `line-length = 120`** and E501 disabled for tests only (long literal contract
   sentences); all source modules obey the limit.
