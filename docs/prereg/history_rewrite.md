# Commit history rewrite, 12 October 2026

Before the first push to GitHub, the local history was rewritten once with `git filter-branch`. Two changes were made:

1. Author and committer were set to the author's GitHub identity, Hsuan Lo <hsuanlo@gsd.harvard.edu>. Earlier commits used another of the author's addresses.
2. Tool-attribution trailers (`Co-Authored-By:` lines) were removed from commit messages, so the author is the sole author.

**Nothing else changed.** Every commit keeps its original author and committer timestamps, and its file tree is byte-identical. For example, the trees of `3a240a6` and `c1bedfb` are equal. All hashes recorded in `docs/prereg/PREREG_SPEC.json` (document and code SHA-256) are content hashes, so they are unaffected. The registration check still passes.

Commit IDs did change. `docs/preregistration.md` (a hashed, unchangeable file) and the append-only `docs/decisions.md` cite the old IDs. Use this table to translate them:

| Old ID | New ID | Commit |
|---|---|---|
| `3a240a6` | `c1bedfb` | AAG abstract: trim to 236 words (239 under strict token counting) |
| `99b18b2` | `18a871a` | PREREGISTRATION: prospective test of anchored, district-adjusted forecasts (2026-2027 origins) |
| `70a99ce` | `7948c2d` | Rank-normalized inputs (secondary); pre-registration document and scoring code |
| `727dac0` | `025cd62` | Secondary (post-freeze): partial-window + anchored recalibration, sensitivity grid |
| `940b887` | `4040f74` | Secondary (post-freeze): ACS tract context (M2e) and rolling recalibration |
| `26ee305` | `56a948c` | Score sealed 2024-25 test once; paper v0.3 with test results, discussion, conclusion |
| `8db7aa6` | `83b83e1` | FREEZE model specification before scoring 2024-25 test origins |
| `94d77a2` | `e768cb3` | Pre-freeze: district-relative warning rule (primary), M2d product model, per-origin top-k |
| `75b4753` | `82cf3c4` | Add M2d (district context) as fifth compared model |
| `33a8dfd` | `28cdee8` | Initial pipeline, research plan, working paper v0.1 and AAG abstract |

The rewrite happened before anything was public, so no published history was altered. The independent timestamp for the pre-registration remains the public registry deposit (`docs/prereg/public_deposit.md`).
