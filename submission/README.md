# CoRL 2026 Workshop submission — "Pretrain to Adapt"

Submission-ready, **anonymized** build of the paper for the CoRL 2026 workshop
*"Pretrain to Adapt: What Makes a Pretrained Policy Adaptable?"*

- Workshop site: https://corl2026-robotcl.github.io/
- Submission guidelines: https://corl2026-robotcl.github.io/call/
- OpenReview: https://openreview.net/group?id=robot-learning.org/CoRL/2026/Workshop/pretrain-to-adapt
- **Deadline: October 4, 2026, 05:00 UTC**
- Track: **full paper, up to 8 pages** (excluding references). A 4-page short
  track also exists if needed.
- **Double-blind** (anonymized) · **non-archival**.

## Files
- `paper.tex` — the anonymized CoRL-formatted paper.

## How to compile
`paper.tex` uses the CoRL style: `\usepackage{corl_2026}` (no `[final]` option,
so the author block is anonymized for review). You need `corl_2026.sty`:

1. **Easiest — Overleaf:** open the official CoRL 2026 template on Overleaf (linked
   from the CoRL 2026 "Instruction for Authors" page), then replace its main
   `.tex` with `paper.tex` from here. The style files are already in that project.
2. **Manual:** download the CoRL 2026 template zip from the author-instructions
   page (Google Drive), copy `corl_2026.sty` (and any files it needs) next to
   `paper.tex`, then `pdflatex paper` ×2.

## Before you submit — checklist
- [ ] **Page count ≤ 8** (excluding references). This paper is a full study with
      four tables; if it runs over 8 pages in the CoRL format, trim (candidates:
      shorten Related Work, merge the Model-A reproduction table into text, tighten
      the probing protocol). Ask and I'll cut it to fit.
- [ ] **Anonymity:** author block is "Anonymous Author(s)"; no name, affiliation,
      email, or repo link anywhere. (Verified: no GitHub/username/affiliation
      strings in the body.) Do **not** upload a PDF whose metadata carries your
      name — the `\hypersetup` here sets only `pdftitle`, no `pdfauthor`.
- [ ] **Honesty items kept:** single-seed (n=1) stated; Wilson CIs + significance;
      objective-vs-corpus confound; sim-only scope. Leave these in — reviewers
      reward them.
- [ ] Compile clean (no undefined refs; all `\ref`s resolve; no `\todo`/`\tbd`).
- [ ] Submit the PDF via the OpenReview link before the deadline.

## Notes
- Non-archival: submitting here does **not** preclude a later, fuller version at a
  main conference (add seeds, Model B, deeper probes) — see the paper's Limitations
  / future-work.
- The non-anonymized, IEEE-formatted version lives at
  `src/openpi/diffusion_backbone/docs/paper.tex` (for arXiv / ICRA-style venues).
