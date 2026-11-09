# Credit union knowledge system

This is my Python submission for the knowledge-base hackathon. It reads the five supplied website snapshots, keeps the history, and answers questions from the right version of the site.

## How it works

- `ingest(snapshot)` stores the pages and builds the search index.
- `answer(question, as_of=None)` finds the relevant information, checks the date and product details, and returns an answer with a source. If the site doesn't support an answer, it says so.
- `conflicts()` finds rate disagreements and records which source takes priority.

The main code is in `solution.py`. I used in-memory storage, a BM25-style keyword index, and direct parsing for the rate tables. I kept the solution dependency-free so it runs with Python 3.11 without any setup beyond the event kit.

## Run it

Put this folder next to `harness/`, `eval/`, and `snapshots_zip/` in the provided kit. From the kit's root directory:

```bash
python harness/run_eval.py --submission hackathon_submission_natural --eval eval/public_eval.jsonl
python hackathon_submission_natural/test_solution.py
```

The last public evaluation score was 0.969, with no execution errors. That score is from the public questions only; the organizers also have hidden questions and a separate conflict evaluation.

The website snapshots aren't included in this submission. Please don't upload or redistribute them.
