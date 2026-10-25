# Evaluation

Everything in `benchmarks/results.json` is produced by one command:

```
uv run python -m clausewise.bench --out benchmarks        # or: make bench
uv run python -m clausewise.bench --out benchmarks --check benchmarks/thresholds.json   # CI gate
```

Environment variables that change the run: `CLAUSEWISE_SEED` (default 42), `CLAUSEWISE_CORPUS_SIZE`
(default 400), `CLAUSEWISE_EMBEDDINGS` (`auto`, `dense`, `tfidf`, `none`). The run takes about one minute
on an Apple Silicon laptop; the QA evaluation dominates.

## Dataset

The corpus is synthetic with complete ground truth. It is generated, not downloaded, so there is no
licence question and every label is exact. The generator is seeded and cached under `data/`.

**Contract types** (weights): NDA 0.20, MSA 0.25, SaaS 0.20, employment 0.20, lease 0.15. With seed 42
and 400 contracts: 75 NDA, 116 MSA, 84 SaaS, 72 employment, 53 lease; 6,809 clauses in total, 17 per
contract on average.

**Clause plans.** Each type has an ordered list of labels with inclusion probabilities
(`corpus/plans.py`). Mandatory labels (probability 1.0) always appear; boilerplate labels (governing law,
disputes, assignment, notices, entire agreement, severability, force majeure) are shuffled with
probability 0.5 so position alone does not identify them. Every contract has a preamble, a signature block
and, with probability 0.85, recitals. 37 labels in total.

**Clause text.** A clause is one alternative from each required sentence slot of its template plus each
optional slot with probability 0.5. Party roles (two naming variants per type), cross references rendered
in the document's own numbering style, and all values are substituted at render time.

**Values.** Durations are drawn from realistic sets (notice 15/30/45/60/90 days, cure 14/30 days, term
12/24/36/60 months, confidentiality survival 2/3/5 years, non-compete 6/12/18/24 months, and so on) and
rendered in one of three styles per contract: `thirty (30) days`, `30 days`, `thirty days`. Amounts follow
the paying party's country (INR with lakh grouping, USD, GBP, EUR, SGD; `INR`, `Rs.`, `US$`, `$`, `S$`
prefixes). Dates use four formats (`12 March 2025`, `March 12, 2025`, `12th day of March, 2025`,
`2025-03-12`). Governing law is drawn from nine jurisdictions with weights India 0.40, England and Wales
0.12, Singapore 0.10, New York 0.10, Delaware 0.08, California 0.06, Germany 0.05, New South Wales 0.05,
DIFC 0.04.

**Planted risks.** Risk-bearing labels draw a variant:

| Label | Variants (probability) | Risk when |
| --- | --- | --- |
| term | fixed 0.45, auto_renew 0.35, hard_negative 0.20 | auto_renew |
| termination | mutual 0.35, for_cause_only 0.15, unilateral 0.30, hard_negative 0.20 | unilateral |
| liability | capped 0.50, unlimited 0.30, hard_negative 0.20 | unlimited |
| ip | retain 0.35, deliverables 0.20, broad 0.30, hard_negative 0.15 | broad |
| non_compete | post_term 0.55, during_term 0.25, hard_negative 0.20 | post_term |
| governing_law | one variant | jurisdiction not in {India, England and Wales, Singapore} |

Hard negatives reuse the risky vocabulary in harmless sentences ("shall not be unlimited and shall be
capped at", "shall not renew automatically", "either party may terminate for convenience", "contains no
covenant not to compete", "nothing in this Agreement shall operate to assign any pre-existing intellectual
property"). With seed 42 the corpus plants 603 risks: 149 governing law, 137 auto-renewal, 94 unilateral
termination, 86 IP assignment, 81 unlimited liability, 56 non-compete.

**Rendering styles** (per contract): numbering `1.` 0.35, `Section 1.` 0.20, `ARTICLE I.` 0.15,
`Clause 1:` 0.10, bare upper-case headings 0.20; inline headings ("3. Term. This Agreement...") with
probability 0.35 for numbered styles; sub-numbered paragraphs ("3.1", "(a)") 0.25; upper-case headings
0.30 (always for ARTICLE and bare styles); hard wrap at 72, 80 or 100 columns or none; "Page n of m"
markers between clauses 0.30.

**Noise.** Half of the contracts are "noisy": each numbered clause loses its heading with probability
0.15 (the clause then starts directly with body text) and 1% of lowercase words of six letters or more get
two adjacent letters swapped. Typos never touch an entity span, so the entity ground truth stays exact;
headings, page markers and the title line are also left alone. Wrapping and typos are length preserving.

**Questions.** Each contract gets up to eight answerable questions drawn from a bank of 39 paraphrased
families (`corpus/questions.py`) plus exactly one unanswerable question about a clause type the contract
lacks. Gold is the clause id and, for value questions, the normalised value (`30 days`, `INR 1250000`,
`2025-03-12`, `India`, `99.9%`, `yes`/`no`, `A | B` for parties, `12 months fees` for fee-based caps).

## Split

Contracts are shuffled with `numpy.random.RandomState(seed)` and 30% are held out: 280 train, 120 test
(62 of them noisy). The classifier is trained on the training split only. Rule-based components
(segmenter, entities, risk rules, answerer, checker) have nothing to train, but they are evaluated on the
same 120 contracts so that every number describes the same documents. The rules and the generator were
written by the same author, so the held-out split protects only the trained model from leakage; see
Threats to validity.

## Metrics

**Segmentation boundary F1.** A predicted clause start matches a gold start when both fall on the same
line of the text. Precision, recall and F1 over all boundaries, reported overall and split by clean and
noisy contracts. **Exact span rate**: share of gold clauses whose (start, end) is reproduced exactly.

**Classification macro-F1.** Unweighted mean of per-label F1 over the 37 labels on the 2,000 test
clauses, using gold segments. Three conditions: model with heading and body features; model trained and
evaluated without headings (body only); the heading-keyword rule baseline.

**Entity F1.** Mention level. A prediction matches a gold mention when type and normalised value are equal
and the spans overlap. Reported per type and overall.

**Risk precision and recall.** A finding matches a planted risk when the risk type is equal and the
clause is the same. On gold segments the clause id is compared directly. End to end, the pipeline
segments the raw text itself and a predicted clause is mapped to the gold clause it overlaps by more than
half; precision and recall are then computed on (type, gold clause) pairs. Per-type counts are in
`results.json`.

**Redline resolution.** For every finding the playbook fallback is rendered with the contract's roles and
the detector is run on the proposed text; the redline resolves the finding when no finding of the same
type remains.

**Retrieval Recall@k and MRR.** For each answerable question the clauses of its contract are ranked by the
backend; Recall@k is the share of questions whose gold clause is in the top k, MRR the mean reciprocal
rank. Backends: BM25, TF-IDF cosine, BM25+TF-IDF fusion, dense (model2vec potion-base-8M), BM25+dense
fusion. The two dense rows are only produced when the model is available; the JSON names the backend.

**QA.** *Citation validity*: among answered questions, the share whose cited clause ids all exist in the
document and whose every sentence passes the grounding checker. *Citation accuracy*: share of answered
questions that cite the gold clause. *Exact match*: on value questions, the normalised short answer equals
the gold value; reported overall and by answer type. *Abstention on unanswerable*: share of the 120
unanswerable questions where the system returned no sentence. *Answered rate* is also recorded.

**Grounding checker.** Positives: the first two sentences of every test clause (3,271 sentences), each
perturbed harmlessly (upper-cased or with one space turned into a newline); the accept rate should be 1.
Negatives (10,039): the first sentence of another clause cited against this one, a number changed, a
negation inserted after the first modal verb or removed when present, and a fabricated sentence. The
reject rate is reported overall and per kind.

**Contract type accuracy.** Share of test contracts whose type is inferred correctly from the title line
and clause labels.

## Thresholds

`benchmarks/thresholds.json` lists the minimum values CI accepts. They sit below the measured values of
both the dense run and the TF-IDF fallback run so that a missing model in CI cannot fail the gate while a
real regression does. The TF-IDF fallback run is recorded in REPORT.md.

## Determinism

The corpus digest (`corpus.sha256` in `results.json`) and every metric are identical across runs with the
same seed and backend; `tests/test_bench.py::test_benchmark_runs_and_is_deterministic` checks this on a
40-contract corpus and REPORT.md records the comparison of two full runs.

## Threats to validity

- The corpus is synthetic. The classifier saturates at macro-F1 1.0 because templates are lexically
  distinct; real contracts paraphrase far more. The heading-rule baseline (0.9454) and the body-only
  condition exist to show the task is not trivial for rules, not to claim the model would score 1.0 on
  real paper.
- The same author wrote the generator and the rules. Risk precision and recall of 1.0 on gold segments
  mean the rules cover every template variant, including the hard negatives; they say nothing about
  wording the templates do not contain.
- Retrieval and QA numbers are the most transferable because the question bank deliberately uses
  vocabulary the clauses lack; even so, the 1,080 questions come from 39 families.
- Abstention on unanswerable questions (0.5833) is the weakest number and is reported as such: a lexical
  answerer cannot know that "rent" is not "fee" when both are monthly amounts.
