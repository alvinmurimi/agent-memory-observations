# Agent memory observations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20787349.svg)](https://doi.org/10.5281/zenodo.20787349)

An exploratory study that began from a practical question, "can structuring a knowledge base make an LLM agent answer more accurately," and is frozen here at observations only.

Start with these two files:

- [OBSERVATIONS.md](OBSERVATIONS.md). What was measured, what held, what failed under controls, and what is explicitly unknown. This is the canonical record.
- [JOURNEY.md](JOURNEY.md). How the study started, what it found, where it over-reached, and why it is paused.

## Summary

Given the relevant evidence inside its context window, a capable LLM reader already resolves coreference, time, and source authority on its own, so structuring that evidence for reasoning did not improve answer accuracy on the tasks tested (stable across two model tiers). The most useful by-product is methodological: LLM reader accuracy showed large run-to-run variance (0.53 to 0.93 on identical inputs), so low-sample single-run comparisons are unreliable. A separate set of synthetic retrieval experiments observed that recall is sensitive to whether a text unit contains an explicit lexical anchor for the queried entity. That is a known information-retrieval phenomenon, measured here in a synthetic regime and never validated against a production pipeline. Nothing here establishes a memory architecture principle.

## Experiments (deterministic, no API keys)

- [recall_ablation.py](recall_ablation.py): recall ablations (E1)
- [e1b_multihop_fair.py](e1b_multihop_fair.py): de-rigged multi-hop re-test (E1b)
- [e1d_recall_boundary.py](e1d_recall_boundary.py): single-shot recall boundary (E1d)
- [e7_scale_retrieval.py](e7_scale_retrieval.py): lexical recall@B vs scale (E7)
- [dense_local.py](dense_local.py): static-dense recall@B vs scale
- [e8_hybrid.py](e8_hybrid.py): lexical + dense RRF hybrid
- [e7b_extraction_curve.py](e7b_extraction_curve.py): write-time locality recovery curve
- [e9_recallK.py](e9_recallK.py): burial-depth / recall@K sweep
- [requirements.txt](requirements.txt), aggregated outputs in [results/](results)

## Reproduce

```
pip install -r requirements.txt
python recall_ablation.py
python e1b_multihop_fair.py
python e1d_recall_boundary.py
python e7_scale_retrieval.py
python dense_local.py
python e8_hybrid.py
python e7b_extraction_curve.py
python e9_recallK.py
```

The LLM-in-the-loop experiments (E2 to E6, run across two model tiers) are not reproducible without a model endpoint; their aggregated metrics are in [results/](results) and their design is described in [OBSERVATIONS.md](OBSERVATIONS.md).

## Related prior work (provenance only)

Listed for lineage. This study does not extend, unify, or refute them.

- [CORE-RT](https://github.com/alvinmurimi/core-rt): a contamination- and tautology-controlled corrective-QA benchmark (read-time resolution and recency effects).
- [StaleRAG](https://github.com/alvinmurimi/StaleRAG): an earlier exploration of temporal staleness in retrieval.

## Status

Paused, on purpose. The useful results are small and engineering-ready. A broader claim would require a deployed-system study that was not done. See [OBSERVATIONS.md](OBSERVATIONS.md) section D for the explicit unknowns.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). Documentation and aggregated results may be reused with attribution.
