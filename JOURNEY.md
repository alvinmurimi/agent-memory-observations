# The journey: from "make agents more accurate" to "observations only"

A candid account of how this study started, what it found, where it over-reached, and why it is paused. Written so the next reader does not repeat the loop.

## Where it started

The original intent was practical: make AI agents give more accurate answers from a knowledge base. The first idea was intuitive and human. Memory should be navigated the way people do it, by entities, events, time, and provenance, an "epistemic swim" over structured memory. The bet was that converting text into structured memory at write time would beat plain retrieval plus reasoning.

## What the earlier work found (and still stands)

Two earlier, separate artifacts ([CORE-RT](https://github.com/alvinmurimi/core-rt) and [StaleRAG](https://github.com/alvinmurimi/StaleRAG)) found that on corrective question answering, read-time deterministic resolution and a recency reranker help, while write-time structuring did not show an accuracy advantage. The bottleneck looked like lossy write-time extraction. Those are modest, controlled, and convergent with recent literature. They are listed here only as provenance; this study does not extend or unify them.

## What this study added (and what holds)

- Inference experiments (E2 to E6). With evidence already in context, a capable reader resolves coreference, time, and authority itself, and structured tags performed about the same as prose. This held across two model tiers. It is solid, but confirmatory. The field is already heading here.
- A genuine methodological by-product. LLM reader accuracy on one task swung between 0.53 and 0.93 on identical inputs across runs. That variance is real, was observed directly, and is the single most useful thing this study produced. It also explained why earlier "effects" kept appearing and dissolving.
- The study retracted two of its own findings (an E1 multi-hop "win" and an E2 "structure hurts" effect) and discarded a rate-limit-corrupted run. The self-correction worked.

## Where it went wrong

After the inference result, the work branched into retrieval ("can the system even find the right memory") and observed a sharp synthetic collapse for text units that lacked an explicit entity anchor. From there the abstraction drifted upward faster than the grounding:

- a measured retrieval sensitivity, became
- "structure is decisive for implicit-reference retrieval," became
- "the epistemic swim splits into read-time and write-time," became
- "write-time memory indexing is the lever."

Each upgrade required a bigger experiment, so it never converged. External review correctly identified the failures. The oracle index is a feasibility ceiling, not a baseline. The collapse is construction-sensitive. The flat case was partly definitional, because the referent was not recoverable. No production retrieval baseline was ever included. And the design is, at best, synthetic manipulation of real data, not a deployed-system evaluation. A retrieval artifact was being read as a structural property of memory. That is exactly the pattern by which retrieval-failure and graph-win papers overclaim.

## Why it is paused (not "finished")

This was a scope-compression failure, not a technical dead end. The useful kernel is small and engineering-ready; the grand version needs a deployed-system study that is a separate project with different methods, and continuing to resolve the theory only reopens the loop. So the work is frozen at observations (see [OBSERVATIONS.md](OBSERVATIONS.md)). It leaves two keepers, the inference result and the variance caution, and one precisely bounded open question: does entity-anchor sensitivity matter in a deployed system. Freezing here returns the original goal, making agents more accurate, to an engineering question answerable by measuring answer correctness in a real pipeline. That is an honest place to stop.
