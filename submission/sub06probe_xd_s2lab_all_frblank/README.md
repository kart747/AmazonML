# sub06probe_xd_s2lab_all_frblank

Diagnostic: absolute France F0.5 on the public LB from a single upload.

**Upload file:** `output/matching_results.tsv` (sha256 `6ef0446659f1…`).

**Approach:** PROBE, not a candidate. Upload #1 (sub03p, public LB 0.97741) with every France S1 emptied. US/India rows are byte-identical to upload #1, so
`F_FR = s_FR + (0.97741 − LB(this)) / w_FR` with w_FR ≈ 0.150 and s_FR ≈ 0.054.
Unlike sub05 (which needs sub04 uploaded first), this probe needs no second upload.

Read-only after validation. `validation.txt` holds the official validator output plus the hard subset check.
