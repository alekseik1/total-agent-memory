# Grounded reader V4, 2026-09-15

Candidate: generate cited premises before the answer; explicitly permit arithmetic
and date operations from those premises; verify relevance and chronology; emit
the verifier's evidence check before its boolean verdict. Exact source quotation,
subject checks, source snapshots, retrieval limits and the five-call cap are unchanged.

Development uses the same 50-question gates per corpus as V3. Baseline answers
and their judgments are reused only after exact context equality is checked.
All questions are reported, including errors and regressions. Runtime receives
questions and retrieved evidence only, never evaluation references.

Freeze the candidate before rerunning the previous 50-question validation sets.
These are repeat public stage-validation sets, not an independent unseen test or
a leaderboard. Report ordinary answers and refusals separately. Do not switch
the default recall path based on aggregate accuracy alone.

Use the existing fixed-model budget client and existing $3 stage ledger.
Before this run: $2.0861216 committed, including eight outstanding reservations;
historical total at most $21.3828064 of the authorized $40. No budget increase.
Baseline artifacts and source databases are read-only; runtime databases are
temporary copies. No live installation, Git write or publication is involved.
