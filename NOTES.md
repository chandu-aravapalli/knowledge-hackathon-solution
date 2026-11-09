# Implementation notes

## Approach

I kept the solution in Python and in memory because the dataset is small enough to load quickly. The scorer gives `ingest()` each snapshot in date order, so there's no web scraping or network access.

For each page, I store the URL, title, timestamp, and a hash of the Markdown content. Unchanged content gets reused instead of indexed again. I split pages into sections and build an inverted index for keyword search. This makes retrieval fast without adding a vector database or an API dependency.

## Picking the right answer

If a question has an `as_of` date, I use the latest snapshot available on or before that date. I also handle dates written directly in the question. If there's no date, I use the September snapshot.

For rates, I read the Markdown tables and match the details that matter: loan term, model year, account type, balance or loan amount, and APR versus APY. Official rate pages take priority over calculators and older articles. I use a simpler search fallback for other questions, but I don't return a number unless the source supports it. If I can't verify an answer, I return `confident=False`.

## Conflicts

`conflicts()` checks for disagreements between comparable facts. It also records published rate changes over time. Different balance tiers, loan terms, and metrics aren't treated as the same fact. Historical changes are labeled as such; they don't automatically mean two pages were wrong at the same time.

## Testing

I ran the provided 40-question evaluation and a separate set of edge-case checks. The public QA score was 0.969, with 44/44 edge checks and 12/12 in-repo unit tests passing in the previous review. These are local results, not a guarantee for the hidden evaluation. The public scorer does not include the organizers' gold conflict list.

## Dependencies and data

Python 3.11 or newer. No third-party packages, paid APIs, API keys, or live website calls. All answers come from the five snapshots in the event kit. The snapshots are not included in this submission and should not be redistributed.
