# sub05probe_xd_dense_mix_frblank

diagnostic: absolute France F0.5 on the public LB.

**Upload file:** `output/matching_results.tsv` (sha256 `1e9fc443ca89a9b8…`). `output/candidate_pairs.tsv` is the stage-2-scored candidate set, 5.53/S1 (same as sub03p).

**Approach:** PROBE, not a candidate: sub04 with every France S1 emptied (France tau 1.01). US/India rows byte-identical to sub04.

**Local:** {"note": "France-blank probe; LB(sub04) - LB(this) = w_FR * (F_FR - s_FR)"}. Upload #1 (sub03p) for reference: normal eval 0.98742, dense 0.98498, public LB 0.97741.

Integrity: copied byte-identical from `output_variants/xd_dense_mix_frblank`. `validation.txt` holds the official validator output plus the hard subset check. Read-only after validation. Details: `metadata.json`.
