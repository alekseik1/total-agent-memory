# Answer-first grounding pilot, 2026-09-15

The user requested continuing release preparation, BGE profiling and QA together.
The V4 prompt-order candidate was rejected for a LoCoMo validation regression.

Hypothesis: generating a natural-language answer before attribution preserves
reasoning that is lost while generating a constrained citation object. This
pilot supplies the frozen baseline answer as an untrusted tentative answer to
the existing grounded reader. It may correct or reject that answer. Exact quotes,
subject binding and semantic verification remain mandatory. No reference answer
or judge label is sent to the reader or verifier.

The first answer is reused from the existing artifacts after exact initial
context equality is checked. Its historical generation cost is already accounted
for. Report this reuse explicitly: this experiment does not measure end-to-end
latency or the cost of a fresh first-answer call. Follow-up retrieval is disabled,
so the proposed pipeline has at most four calls including the reused generation
(one natural answer, up to two grounding attempts, one verification).

Evaluate all50 gate questions per corpus. If promising, freeze the candidate
before repeating all50 validation questions per corpus. Public repeated sets are
not independent unseen validation. Report ordinary answers, refusals, regressions
and errors separately. Do not enable this in production until implemented and
verified as an actual bounded pipeline; this pilot is research code only.

New stage ledger cap $1.50, within the user's existing $40 total authorization.
Previous committed upper bound $22.1175536; even a fully spent stage remains
below $23.6175536. The old $3 ledger is not changed. Frozen databases are read-only;
evaluation uses temporary copies. No live data or deployment changes.
