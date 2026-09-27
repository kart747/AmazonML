# sub04_xd_dense_mix

upload #2 candidate: single-factor test of the density/orphan hypothesis vs upload #1 (same candidates, same features).

**Upload file:** `output/matching_results.tsv` (sha256 `4b4b064ccc1b2a57…`). `output/candidate_pairs.tsv` is the stage-2-scored candidate set, 5.53/S1 (same as sub03p).

**Approach:** xd candidates + stage2_lab `all` features, stage 2 retrained on normal + dense-world rows (density_train --mix: 19% of train S1 dropped, their copies become orphans, context recomputed), early stopping on the dense tune fold; thr+excl τ 0.68 (dense tune).

**Local:** {"normal_tune": 0.98768, "normal_eval_fold0": 0.98743, "dense_tune": 0.98634, "dense_eval_fold0_at_normal_tau": 0.98625, "density_loss": -0.0012, "normal_fp": 2162, "dense_fp": 2542}. Upload #1 (sub03p) for reference: normal eval 0.98742, dense 0.98498, public LB 0.97741.

Integrity: copied byte-identical from `output_variants/xd_dense_mix`. `validation.txt` holds the official validator output plus the hard subset check. Read-only after validation. Details: `metadata.json`.
