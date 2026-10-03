# sub08_xd_s2lab_all_veto

Upload #1 (public LB 0.97741, sha f9ff0a26696c) with 4,667 kept pairs removed and nothing added:
pairs whose target name is the S1 name plus exactly one affiliate/branch token (group, holdings,
ventures, industries, northside, midtown, ...). These tokens were learned on train tune fold 3,
where single-extra pairs with them are at most 2% true. US 1,840 and India 2,826 pairs are removed,
plus 1 France pair. The candidate file is byte-identical to upload #1.

Differs from sub07 by dropping the France same-street/different-house-number guard (frhn), which
E-G9 showed is not supported. Upload `output/matching_results.tsv`. See `metadata.json` for the evidence.
