# STORM Implementation Status

This file explains why only the first module was added, what is missing, and
what is needed next to proceed safely.

## What is implemented now
- **GROBID-based methods extraction (scaffolding only).**
- **Database schema helper** for upcoming research tables.
- **Opt-in hook** in `storm_commander.py` controlled by
  `STORM_ENABLE_METHODS_EXTRACT=1`.
- **Phase-1 extraction toggles** for citations, tables, figures, and OpenAlex.

## Why the rest is not added yet
1) **Source-heavy modules need clear access rules first.**
   - OpenAlex, S2ORC, and other sources require explicit limits (rate, quota,
     storage, and legal boundaries) before integrating.
2) **Large data modules need storage and CPU budgets.**
   - DeepFigures, TableBank, and S2ORC can quickly consume disk and compute.
3) **Quality control must be defined before mass ingestion.**
   - Adding everything at once can flood the system with noise and make it hard
     to trace failures or validate outputs.

## What is needed from you next
Please confirm the following so the next modules can be added safely:
1) **Data limits**
   - Maximum PDFs/day to ingest.
   - Max disk space the system can use.
2) **Source priority**
   - Order of sources: OpenAlex, S2ORC, Sci-Hub, etc.
3) **Language priority**
   - Turkish-first, English-first, or balanced.
4) **Quality filter**
   - Minimum criteria for a paper to be kept (citations, venue, year, etc.).
5) **Telegram approval workflow**
   - Exact approval keywords and whether rejection should stop or just skip.

## Proposed next steps (Phase 1 continuation)
1) **OpenAlex ingestion adapter** (metadata + open access PDF links).
2) **ParsCit citation extractor** for structured references.
3) **Tabula for tables** (limited to a small sample set).
4) **deepfigures-open for figures** (optional, after storage limits defined).

Once those are confirmed, the remaining modules can be layered on top safely.