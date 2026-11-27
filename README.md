# clausewise

[![ci](https://github.com/raunitgrey7/clausewise/actions/workflows/ci.yml/badge.svg)](https://github.com/raunitgrey7/clausewise/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Contract intelligence that works offline and explains itself. Paste an NDA, MSA, SaaS, employment or
lease agreement and clausewise segments it into clauses, labels each clause, extracts the parties,
dates, amounts, durations and governing law, scores the document against a risk playbook with the exact
sentence that triggered each finding, proposes redlines, and answers questions with sentences copied
verbatim from the contract and cited by clause id. A grounding checker rejects any answer sentence a
cited clause does not support, so the system abstains rather than invents.

Everything runs on a laptop CPU with no API key: rules, a scikit-learn classifier, BM25 plus a 30 MB
static embedding model, and an extractive answerer. An Anthropic model can be switched on behind a flag
for the answering step; it goes through the same grounding checker.

## Why

Contract review tools either match keywords (and flag "liability shall not be unlimited" as unlimited
liability) or hand the document to a language model (and cannot prove the answer came from the text).
clausewise takes a third route: deterministic components with measured accuracy on a corpus with known
ground truth, hard negatives built into the benchmark, and a checker that turns "grounded" from a claim
into a test.

## Architecture

```
 text ──> segment ──> classify ──┐
             │                   ├──> risk playbook ──> redlines ──┐
             ├──> entities ──────┘                                  ├──> AnalysisResult
             │                                                      │
             └──> retrieval (BM25 + dense, RRF) ──> extractive QA ──> grounding checker ──> Answer
                                                        ^
                                              optional LLM (same checker)

 FastAPI: /api/analyze  /api/ask  /api/samples  /health  /metrics  / (dashboard)
```

Modules and data flow are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```
make setup      # uv sync --all-extras, then download the 30 MB embedding model once (falls back to TF-IDF)
make test       # 215 tests: unit, integration, property-based, adversarial, determinism
make bench      # generate the 400-contract corpus, evaluate every stage, write benchmarks/results.json
make serve      # API and dashboard on http://localhost:8000
uv run clausewise analyze path/to/contract.txt     # or: uv run clausewise ask contract.txt "Which law governs?"
```

Python 3.13 and [uv](https://docs.astral.sh/uv/) are the only prerequisites. `docker build -t clausewise .`
then `docker run -p 8000:8000 clausewise` gives the same service as a non-root container with the model baked in.

## Measured results

Every number below is written by `make bench` into `benchmarks/results.json` and copied here by
`scripts/update_readme.py`; CI fails if the two disagree. Methodology, metric definitions and the corpus
specification are in [docs/EVALUATION.md](docs/EVALUATION.md); the exact commands and outputs are in
[REPORT.md](REPORT.md).

<!-- results:start -->
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
<!-- results:end -->

How to read the table:

- Retrieval and QA are evaluated per contract: the question is asked about one document and the backend
  ranks that document's clauses. Fusing BM25 with the dense model lifts Recall@5 from 0.8177 to 0.9146.
- Citation validity is 1.0 by construction for the extractive path; it is reported because the same
  metric is what the optional LLM path has to hold.
- Classification saturates on this corpus (templates are lexically distinct). The heading-rule baseline
  at 0.9454 shows the labels are not trivial for rules, but the 1.0 should not be read as a claim about
  real contracts.
- Abstention on unanswerable questions (0.5833) is the weakest number. A lexical answerer cannot tell
  that a question about rent should not be answered from a fee clause.
- Half of the test contracts are noisy (dropped headings, letter swaps); the clean/noisy split for
  segmentation is in the table, and risk detection is reported both on gold segments and end to end.

## API

| Route | Purpose |
| --- | --- |
| `POST /api/analyze` `{text}` | clauses with labels and confidences, entities, risk findings with evidence and playbook position, redlines with diffs, risk score |
| `POST /api/ask` `{text, question, mode?, llm?}` | grounded answer: verbatim sentences with clause ids, short answer, rejected sentences, retrieved ids |
| `GET /api/samples`, `GET /api/samples/{id}` | sample contracts from the seeded corpus for the dashboard |
| `GET /health` | version, embedding backend in use, auth and LLM flags |
| `GET /metrics` | Prometheus text format: request counts and latencies, analyses, findings by type, answers by outcome, grounding rejections |
| `GET /` | single-page dashboard (vanilla JS, no build step) |

Every response carries an `X-Request-ID` header (yours if you send one) and every log line is JSON with
that id. Set `APP_API_KEY` to require an `X-API-Key` header on the `/api` routes; `/health` and `/metrics`
stay open for probes.

### Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `CLAUSEWISE_SEED` | `42` | corpus and model seed |
| `CLAUSEWISE_CORPUS_SIZE` | `400` | contracts generated for training and the benchmark |
| `CLAUSEWISE_EMBEDDINGS` | `auto` | `auto` uses the cached model or downloads it once; `dense` requires the cache; `tfidf` or `none` disables dense retrieval |
| `CLAUSEWISE_EMBED_MODEL` | `minishlab/potion-base-8M` | model2vec checkpoint |
| `CLAUSEWISE_DATA_DIR` | `./data` | corpus and classifier cache |
| `CLAUSEWISE_GROUNDING_THRESHOLD` | `0.85` | ordered-token match ratio required by the checker |
| `CLAUSEWISE_MAX_TEXT_CHARS` | `400000` | request size limit |
| `APP_API_KEY` | unset | enables API-key auth |
| `CLAUSEWISE_LLM`, `CLAUSEWISE_LLM_MODEL`, `ANTHROPIC_API_KEY` | off, `claude-haiku-5-5` | optional LLM answerer (`uv sync --extra llm`), used only when a request sets `llm: true` |

## Limitations

- The corpus is synthetic. Numbers measure coverage of the generator's templates, hard negatives and
  noise, not performance on real contracts. The generator and the rules have the same author.
- English only, plain text only. PDF and DOCX extraction is out of scope; wrapped lines and page markers
  are handled because extracted text contains them.
- The answerer is lexical. Paraphrases that share no vocabulary with the clause (even through the synonym
  table) are abstained on, and some duration questions pick a neighbouring duration (exact match on
  duration questions is 0.8632).
- The grounding checker verifies support at the sentence level with exact numbers and negation counts.
  It cannot detect a sentence that is verbatim but answers a different question.
- The playbook is one reviewer position per risk type. Real playbooks depend on which side you are on.

## Roadmap

1. Evaluate on a public contract dataset with clause labels and report the transfer gap.
2. A learned re-ranker for QA candidates, trained on the question bank, to replace the hand-written
   synonym table.
3. Entailment-based grounding for the LLM path, with the current checker kept as the hard gate.
4. Side-aware playbooks (reviewing as supplier or customer) and clause-level negotiation history.

## Layout

```
src/clausewise/        pipeline modules (see docs/ARCHITECTURE.md)
src/clausewise/corpus  generator, templates, plans, values, question bank
src/clausewise/api     FastAPI app, metrics, static dashboard
tests/                 215 tests
benchmarks/            results.json, results.md, thresholds.json
docs/                  ARCHITECTURE.md, EVALUATION.md
scripts/               quality gates and README sync
```
