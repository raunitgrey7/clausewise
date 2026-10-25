# Architecture

clausewise is a deterministic pipeline with one optional LLM step at the very end. Every stage is a
plain Python module with a narrow interface, and the same `Analyzer` object drives the CLI, the API and
the benchmark, so a number measured by `make bench` describes the code that serves `/api/analyze`.

## Data flow

```
                     contract text (str)
                            |
                            v
 +---------------------------------------------------------------+
 | segment.py          line classifier -> clause spans           |
 |                     (numbered, Section n., Clause n:, ARTICLE,|
 |                      bare upper-case titles, unheaded fallback)|
 +---------------------------------------------------------------+
        |                    |                     |
        v                    v                     v
 classify.py            entities.py            risk.py
 TF-IDF(body, heading)  regex + normalisation  sentence-scoped rules
 + dense cues           durations, money,      positives and vetoes,
 -> LogisticRegression  dates, percentages,    governing law vs playbook
 -> label, confidence   jurisdiction, parties  -> RiskFinding + evidence
        |                                           |
        |                                           v
        |                                      redline.py
        |                                      playbook fallback text,
        |                                      unified diff, rationale
        v
 retrieval.py  BM25 | TF-IDF | dense (model2vec) | RRF hybrid, per document
        |
        v
 qa.py         IDF-weighted sentence scoring, typed candidates, yes/no via
               risk rules, parties via preamble -> verbatim sentence + id
        |
        v
 grounding.py  verbatim or ordered-token match, exact numbers, negation
               count inside the aligned span -> accept or reject
        |                                    ^
        v                                    |
 Answer (sentences, citations, short answer)  llm.py (optional, same gate)
```

The API (`api/app.py`) wraps the pipeline in FastAPI with request ids, JSON logs, Prometheus metrics and
optional API-key auth; the dashboard (`api/static/`) is vanilla JS served from the same process.

## Modules

| Module | Responsibility | Why it is shaped this way |
| --- | --- | --- |
| `corpus/clause_templates.py` | Sentence pools per clause label and variant | Hard-negative variants reuse risky vocabulary so keyword matching is punished in the benchmark |
| `corpus/plans.py` | Which labels each contract type contains, inclusion probabilities, variant mixes | This file is the statistical specification of the corpus |
| `corpus/values.py` | Typed placeholder values with normalised ground truth | Entity ground truth is exact by construction, never re-derived by regex |
| `corpus/generator.py` | Seeded assembly and rendering with offset tracking | Wrapping turns spaces into newlines so no character offset ever moves; typos avoid entity spans |
| `corpus/questions.py` | Paraphrased question bank with gold labels and values | Paraphrases use words absent from the clauses so dense retrieval has something to add |
| `textutil.py` | Tokeniser, stemmer, sentence splitter, number words | One token definition shared by BM25, the answerer and the grounding checker keeps the extractive path self-consistent |
| `segment.py` | Clause segmentation | Line based: inline headings and sub-numbering fall out of the same rules, offsets stay exact |
| `classify.py` | Clause classifier and heading-rule baseline | Dense structural cues keep the model working when headings are missing or generic |
| `entities.py` | Entity extraction and normalisation | Context-bound jurisdiction pattern avoids "incorporated under the laws of" in preambles |
| `playbook.py` | Reviewer positions, severities, fallback language | Data, not code: a legal team can change a position without touching detection |
| `risk.py` | Rule engine | Sentence scope plus vetoes is what separates "shall not be unlimited" from an uncapped clause |
| `redline.py` | Replacement text and diff | The benchmark checks the loop closes: the proposal must clear the detector |
| `embeddings.py` | model2vec backend with logged TF-IDF fallback | A benchmark number is always labelled with the backend that produced it |
| `retrieval.py` | BM25, TF-IDF, dense, RRF | Reciprocal rank fusion avoids mixing score scales |
| `qa.py` | Extractive answerer | Never writes its own sentence; abstains instead of guessing |
| `grounding.py` | Support checker | Exact numbers and negation counts are what paraphrases get subtly wrong |
| `llm.py` | Optional Anthropic answerer | Same checker, so the LLM can only ever add recall, not hallucinations |
| `pipeline.py` | `Analyzer`: training, caching, `analyze` and `ask` | One code path for CLI, API and benchmark |
| `bench.py` | Evaluation harness and quality gate | Writes `results.json` and `results.md`; `--check` compares to thresholds |
| `api/` | FastAPI app, metrics, static dashboard | Lazy state so tests build apps cheaply; `/health` and `/metrics` stay open |
| `cli.py` | `clausewise generate|analyze|ask|bench|serve|prefetch-embeddings` | Thin wrappers over the same objects |

## Key design decisions

1. **Offsets are sacred.** The generator records every value's span while rendering, and every later
   transformation (wrapping, typo injection) is length preserving. The segmenter, the entity extractor and
   the benchmark all work in character offsets of the original text, so there is no alignment step and no
   place for off-by-one errors to hide.
2. **Rules work on sentences, not clauses.** A risk fires when one sentence matches a positive pattern
   and none of the vetoes. Clause-level matching cannot tell "liability shall not be unlimited and shall be
   capped at X" from "liability shall be unlimited".
3. **Retrieval ranks, the sentence scorer decides.** Documents have a few dozen clauses, so the answerer
   scores every sentence with an IDF-weighted overlap computed on the document itself and uses the
   retrieval rank only as a small penalty. Questions that ask for a value are restricted to sentences that
   carry a value of that type, which is what makes abstention possible.
4. **Grounding is a gate, not a score.** An answer sentence is either verbatim, or an ordered token match
   with every number present and the same number of negations inside the aligned clause span, or it is
   rejected. The LLM path goes through the identical function.
5. **Deterministic by construction.** Every random draw comes from `random.Random(seed * 1_000_003 + i)`
   or a seeded numpy RandomState; scikit-learn gets a fixed `random_state`; model2vec embeddings are a
   static lookup. Two benchmark runs with the same seed produce identical metrics (checked in
   `tests/test_bench.py` and in REPORT.md).

## Extension points

- **Playbook**: `Playbook.from_json(path)` overrides acceptable jurisdictions and per-risk positions,
  severities, weights and fallback text.
- **Embeddings**: any object with `name` and `encode(texts) -> unit-norm array` satisfies the `Embedder`
  protocol; `CLAUSEWISE_EMBED_MODEL` selects another model2vec checkpoint.
- **LLM**: `LLMAnswerer` accepts any object exposing `create(**kwargs)` with the Anthropic Messages shape,
  which is how the tests drive it without network access.
- **New clause types or risks**: add templates and a variant mix in `corpus/`, a rule in `risk.py`, a
  position in `playbook.py`; the benchmark picks them up automatically because ground truth comes from the
  generator.
