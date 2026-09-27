# sub07_xd_s2lab_all_guard

Candidate and diagnostic in one upload: upload #1 (sub03p, public LB 0.97741) minus two post-hoc precision guards (`src/experiments/lb_guards.py`, EXPERIMENTS E-G8). Candidates are byte-identical to upload #1, and the scores are the same saved scores.

**Upload file:** `output/matching_results.tsv` (sha256 `c2646be519ec…`).

- **veto** (US 1,840 + India 2,826 pairs removed). A kept pair is dropped when its only name difference is one extra target token from the generator's affiliate/branch set. The set (group, holdings, ventures, industries, southside, midtown, …, 46 tokens) is learned on tune fold 3, where those pairs are ≤2% true. On eval fold 0 the veto removes 49 pairs, 0 of them true (F 0.98742 → 0.98748). Test keeps these at 20–40× the eval rate.
- **frhn** (France only, 9,140 pairs removed from 6,319 S1). A kept France pair is dropped when the S1 and the target name the same street with a different house number. France true copies almost never change the number: 0.39% at p ≥ 0.999, against US 6.9% and India 2.6%. Kept France pairs differ in 1.27% of cases, so about 63% of these are planted look-alikes. Label-free; the rule would hurt US/India and is not applied there.

**Predicted Δ public LB vs 0.97741:** veto +0.0007; frhn +0.0006 at a 63% planted share (break-even 37%, +0.0009 at 80%). Total ≈ +0.0013. Reading: ΔLB − 0.0007 ≈ the frhn part, which gives France's planted share.

Read-only after validation. `validation.txt` holds the official validator output plus the hard subset check.
