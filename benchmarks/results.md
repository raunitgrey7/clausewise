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
