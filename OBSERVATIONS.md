# Observations: a measurement log with boundaries

Companions: [README.md](README.md) (overview), [JOURNEY.md](JOURNEY.md) (how this unfolded and where it over-reached).

This is the frozen, canonical record of this study. It states only what was measured, what held, what failed under controls, and what is explicitly unknown. It contains no architecture conclusions, no memory hypothesis, and no systems claim.

Scope note. All results below are on synthetic data or synthetically manipulated text, with simplified retrievers (TF-IDF, a lightweight static dense embedder, an RRF hybrid). None of it was validated against a production retrieval pipeline (deployed chunking, embedding, reranker, metadata filters), and none of it measured end-to-end answer correctness in a real agent. Treat every number as the behavior of this setup, not a property of memory systems in general.

## A. What was measured, and was stable

1. Reader, evidence in context (experiments E2 to E6). When the decisive evidence is placed in the reader's context, a capable LLM answers as-of temporal selection, abbreviation, role-reference, and pronoun coreference, and source-authority selection at ceiling (about 0.9 to 1.0), and structured tags perform about the same as plain prose within noise. This held across two model tiers (a frontier reader and a smaller one). Example: as-of selection, a single-run two-by-two design at n=50, gave TERSE_ASOF 1.00, TERSE_NL 0.88, CLAR_ASOF 0.94, CLAR_NL 0.94. Coreference minimal pairs (E4, E5, E6): named, role-reference, and pronoun forms all scored about 0.96 to 1.00, with zero wrong-entity mis-resolution.

2. Single-shot lexical recall boundary (E1d, deterministic, no LLM). Recall of the decisive note was 1.00 when it shared at least one distinctive token (entity or attribute) with the query, and 0.00 when it shared none. A neighbor-expansion (document-locality) step recovered the token-disjoint case to 1.00. Reproducible exactly.

## B. What failed or did not survive controls (retracted)

3. "Iteration is the only recall lever" (an E1 multi-hop claim). Did not survive a de-rigged re-test. It was an artifact of an eight-predicate vocabulary (token collision). Retracted.

4. "Structured tags hurt as-of selection" (E2, an apparent 0.94 versus 0.50 gap). Did not survive. It traced to an under-instructed reader plus large LLM run-to-run variance: the same condition scored 0.53 and 0.93 on identical inputs across separate runs. Retracted as an effect. The variance itself is the finding, and it means single-run, low-sample LLM comparisons are unreliable: use within-run, powered designs.

5. "Implicit reference is unfindable at scale" (the flat version of E7). Partly definitional. In that construction the referent had no recoverable link anywhere in the corpus, so the intended answer was undetermined and the collapse was closer to a tautology than a retrieval failure.

## C. What was measured in the retrieval setups (bounded, not a memory claim)

Under the specific simplified retrievers tested, recall@B of a decisive text unit is sensitive to whether that unit contains an explicit lexical anchor for the queried entity.

- Named units (entity present): lexical, static-dense, and RRF-hybrid all scored 1.00 up to a corpus of 50,000.
- Anchor-removed units (entity replaced by a generic reference, synthetic): recall fell toward 0 as the corpus grew, for all three retrievers. The unit's rank sat in the top few percent but below practical reranker windows, and the window needed to reach it grew roughly linearly with corpus size (median rank about 343 at 10,000 and about 2,224 at 50,000).
- A cheap write-time locality heuristic recovered the synthetic gap on unambiguous sessions (1.00), and about 0.52 when a confounder intervened. An oracle index scored 1.00, but an oracle is a feasibility ceiling, not a baseline.

This is a known class of information-retrieval effect (entity-centric retrieval, document-fragmentation sensitivity), reproduced in a synthetic regime. It is not evidence about deployed systems.

## D. What is explicitly unknown (missing baselines, out of scope)

- No production retrieval baseline (real chunking, a deployed embedding model, a reranker, metadata filters). Everything here is simplified retrieval versus manipulated text.
- Whether the anchor sensitivity holds on real corpora or deployed systems. Untested. A real-log attempt was abandoned for privacy, and even that would have been controlled manipulation of real data rather than a deployed evaluation.
- Whether any of this changes end-to-end answer correctness in a real agent pipeline. Never measured.
- The discourse-position versus reference-form confound. Not separated.

Nothing above implies a memory architecture, a write-time necessity, or a systems recommendation. Those would require a deployed-system study that was not done.
