# Research & Experiment Design — Amazon ML Challenge 2026 (Business Entity Resolution)

**Status:** research phase, revision R1 (2026-09-26 19:30 IST). No pipeline code written yet.
**Deadline:** 27 Sep 2026 23:59 IST (≈ 28 h from R1). Max 5 leaderboard submissions per day.
**How this document grows:** each new research batch gets a row in §2.4 and a delta entry in §10. Inventory (§3) and experiment matrix (§5) rows are updated in place, never duplicated.

### Evidence grades used throughout
| Grade | Meaning |
|---|---|
| **[M]** | Measured by us on the real data (scripts in session scratchpad; reproducible) |
| **[R]** | Measured by a repo on full data with a sound protocol (S1-grouped split, full target pool, exact metric) |
| **[S]** | Self-reported by a repo on a small sample or unclear protocol. Directionally useful, level not trusted |
| **[C]** | Claim without measurement (README prose, AI-generated research reports) |

---

## 1. Challenge understanding

### 1.1 Data (all [M])
| | S1 | S2 | S3 | GT rows |
|---|---|---|---|---|
| Train | 2,206,821 (US 1,323,633 · India 883,188) | 5,034,616 | 5,285,603 | 2,206,821 |
| Test | 1,732,544 (India 809,986 · US 663,106 · **France 259,452 = 15.0%**) | 4,887,273 | 5,082,316 | — |

- All 7 files parse with exactly 4 (or 2) tab-separated fields per row. **6 rows in train S2 contain `"`**, so always read with `quoting=csv.QUOTE_NONE`.
- No empty names or countries. Empty target addresses: S2 3.4%, S3 3.3% (train). S1 addresses are never empty.
- Every GT S1 id is in S1 and vice versa. Every GT target id exists in S2/S3.

### 1.2 Ground-truth structure (all [M])
| Fact | Value | Design consequence |
|---|---|---|
| Singletons (no match) | **5.58%** | The predict-nothing baseline scores only 0.056. This is a *recall-hungry* problem despite F0.5; "be very conservative" is capped low |
| Matches per S1 | mean 3.46; mode 3; max 11; 72% have ≥3 | Decoder must output variable-size sets. Never top-1 |
| True pairs | 7,638,365 (S2 3.69M, S3 3.94M) | ~0.5 S2 and ~0.5 S3 share per S1. S2/S3 each hold *several copies* of the entity |
| **Targets with >1 S1 owner** | **0 of 7.64M** | Constraint is **many-to-one** (each target ≤1 S1; an S1 has many targets). *Not* one-to-one, so Hungarian assignment is the wrong tool |
| **Cross-country true pairs** | **0** | Partitioning by country label is lossless on train. Label-agnostic, so it works for France |
| Unmatched ("distractor") targets | S2 26.6%, S3 25.4% | Hard negatives exist naturally in the pool |
| Test targets per S1 | train 4.68 → test 5.75 (+23%) | Test has a higher distractor density, so thresholds tuned on train are optimistic (see §6.3) |

### 1.3 Signal available in true pairs (200k sampled true pairs, [M])
| Signal | Share of true pairs |
|---|---|
| Exact raw name equal | 4.7% |
| Exact normalized name equal (lowercase, accents folded, punctuation stripped) | 25.7% |
| Share ≥1 normalized name token | 85.5% |
| Share ≥1 address token | 95.6% |
| Exact normalized address | 8.4% |
| Share ≥1 number in address | 79.8% |
| Name script differs (S1 always Latin; target Indic) | 7.3% |
| S1 names shared with ≥1 other S1 | **39.3%** (1.52M distinct of 2.21M) |

The consequence: **name alone cannot identify an entity** (39% of S1 names are non-unique), and **exact-key blocking is hopeless** (only 26% of true pairs share an exact normalized name). Address carries the decision in ambiguous cases.

Noise observed in samples [M] (matches Ayan's catalogue [R]): typos/leetspeak (`C0astal`, `Flraaklin`), token reorder, legal suffix moved or bracketed (`[LLC] Coastal Tungsten`), truncation (`Krimson Energy`), domain names (`coastaltungsten.com`), `M/s` prefixes, full Indic-script names and state names (`दिल्ली`, `उत्तर प्रदेश`), house-number perturbation (`2454`→`245`, `1014`→`014`), city swaps (Hillsboro→Beaverton), `H.NO`/`#`/`NO.` prefixes, component reordering.

France (test only, [M] from samples): legal forms SARL/SAS/SCI/EURL, accents injected (`Fîls`, `Çulturelle`), street abbreviations `R.`/`AV.`/`BD`, region↔city reorder, uppercase S2 addresses, empty S3 addresses. Ayan [R] adds: region↔département swaps and very repetitive names built from a small vocabulary.

### 1.4 Metric implications
- Per-S1 F0.5, macro-averaged over all test S1, singletons included.
- For an S1 with *n* true matches, predicting *k* correct plus *f* wrong: one FP on a 1-match entity costs more than one FN. **But 94% of S1 have matches**, so recall still matters a lot.
- Decision theory: with calibrated p, the optimal plug-in threshold is ≈ F*/(1+β²) ≈ 0.78 when F* ≈ 0.98 [R, Ayan]. Consistent with measured optima of 0.7–0.8 [R] and 0.75–0.86 [S] across repos.
- The **public LB is a subset of test**; private decides. Never tune to public.

### 1.5 Competition constraints (official)
- Output `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the exact set the model scores; every match must be a candidate).
- Validate with `student_resource/utils/validate_submission.py` before every upload.
- Final model: MIT/Apache-2.0 license, ≤8B params. **No external lookup** (APIs, geocoding, registries, internet augmentation).
- Zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, filled `Documentation_template.md`.
- Plagiarism leads to disqualification. See §2.5.

### 1.6 Compute reality
| Machine | CPU | RAM | GPU | Status |
|---|---|---|---|---|
| This machine | 16 threads | 19 GB (≈12 GB free) | RTX 3050 Laptop **4 GB** | verified |
| 3050 #2, 4050, 5060 | ? | **?** | as stated | **unknown RAM/CPU — needed for allocation (§7)** |

Missing Python packages here: polars, pyarrow, duckdb, lightgbm, rapidfuzz, sparse_dot_topn (pandas, numpy, scipy, sklearn and torch are present). Loading S1+S2+S3 train with pandas takes 3 min and 7.5 GB RSS [M].

**Key compute finding:** every strong solution found is **CPU/RAM-bound**. Retrieval (sparse matmul), string features (RapidFuzz) and GBDT dominate. GPUs are useful only for optional neural stages. The four machines matter mainly as four CPU+RAM pools.

---

## 2. Open-source findings

### 2.1 Tier 1: rigorous, full-scale evidence
**AyanAhmedKhan/amazon-ml-challenge** (created 25 Sep, no license, all-rights-reserved) [R]
- **Pipeline:** learned normalization → 5-view sparse TF-IDF blocking per country incl. **reverse target→S1** → 87 features → 3-fold cross-fit LightGBM → OOF collective/sibling features → stage-2 LightGBM → threshold + exclusivity.
- **Measured on fold 0 (220,683 S1, full pool, exact metric, bootstrap SE 0.00016):**

| System | Macro F0.5 |
|---|---|
| Best no-ML rule | 0.767 |
| LightGBM, thr 0.7 + excl | 0.9846 |
| 3-fold cross-fit | 0.9849 |
| **+ stacking with collective features** | **0.9870** (+0.0021, ≈13 SE) |
| Blocking oracle | 0.9965–0.9972 |

- **Blocking (full train pool):** name-only R@40 **0.677** (same-name flooding); address R@40 0.886; combo (0.5 name + 0.5 addr) R@40 0.975; **reverse R@1 0.972, R@8 0.9875**. Union pair recall 0.9905 at 112 cands/S1. Pruned to rev8 ∪ combo10 ∪ name5 ∪ cat5 ∪ addr5 it gives ≈0.989 at ≈45 cands/S1.
- **Negative results (valuable):** expected-F0.5 DP 0.98457 and π-hurdle 0.98466 do **not** beat a tuned threshold 0.98492. Hard exclusivity adds ≈0 once competition features exist. Per-country feature standardization hurts (−0.011).
- **Density:** simulating test density moves the optimal τ 0.7→0.8 and costs −0.0015. Training at test density recovers +0.0014.
- **Unseen country (LOCO):** US→India 0.945 (−0.040), India→US 0.974 (−0.011). Self-training helps only with ≥99% clean pseudo-labels.
- **Top features:** many-to-one *competition* features (`c_combo_tmargin`, `trank`) dominate importance.
- **Errors:** FPs are *planted sibling distractors* (same name, same street, nearby house number). FNs are mostly *undecidable* (empty address + generic shared name).
- **Cost:** 32 vCPU / 128 GB cloud (Modal). Blocking took 2h10m on train. 88M test pairs. Stopped by a spend limit on 25 Sep.
- **Caveats they flag themselves:** learned maps include validation-fold pairs (mild leak). The stage-2 gain used a feature that may not transfer (fixed later).

**jvikramsrd/business_entity_resolution** (created 25 Sep, no license) [S for scores, R for methodology]
- **Pipeline:** strict ingest → normalization with 9-script Indic transliteration → **inverted-index key blocking** with caps and conjunctions → LightGBM → isotonic calibration → expected-F0.5 decoding with exclusivity normalization.
- **Most important finding:** 5% samples overstate full-scale recall by **5 points** (99.35% vs 94.4%). Key-based blocking on full US reached only 0.944 pair recall.
- **Decoder ablation (5% sample):** expected-F ≈ tuned threshold (tie). Exclusivity normalization gives a small gain, with CI excluding 0. Untuned 0.5 is worse (−0.0009).
- **Engineering ideas:** stage manifests with fingerprints (resume), RAM-budgeted chunking, two-machine sharding by country. Runs on a 16 GB laptop, but a full run did not finish there.

### 2.2 Tier 2: challenge-window repos with partial evidence
| Repo (created) | Approach | Reported | Grade and notes |
|---|---|---|---|
| **TanmayJaiswal-IITBH/crossscript-entity-resolution** (26 Sep) | Per-field (name/addr) cosine retrieval, union → 52 feats LightGBM, isotonic, relative per-entity features | Blocking 0.894 → 0.945 recall. **Macro F0.5 0.936** on 40k val vs full pool | [R-ish] honest. Shows key-based and one-field retrieval plateau far below TF-IDF + reverse. "Predict-nothing floor is 0.056." Rule `score ≥ τ and ≥ best − δ` |
| **Nikhil-iitg27/AmazonML2026** (25 Sep) | Indic → Latin via Unicode block-offset alignment, 3 TF-IDF char-n-gram channels, 40 feats, per-source caps, global exclusivity | Recall 0.982 at 65.6 cands. **0.967** on 20k "pessimistic" benchmark | [S] Address is the strongest single channel (0.909 vs 0.749 name) |
| **Akash-bardia/amazon-ml-challenge-2026** (25 Sep) | Multi-attribute inverted index, 28 feats LightGBM, consistency and exclusivity, source-specific τ (S2 0.75, S3 0.80–0.85) | Recall 0.965 at 14.5 cands. **0.976** on 5k val. Exclusivity +0.0007, source τ +0.0003 | [S] small val. <2.2 GB RAM, 58 min full test |
| **Raj-Kanna-05/Amaz_ML** (26 Sep) | Char-3gram TF-IDF inverted index (stop-ngrams >3%, postings cap 800) + address view, LightGBM, GroupKFold, target conflict → argmax | 0.989 recall on **2k** entities. LOCO India→US 0.991, US→India 0.943 | [S] tiny benchmark. Useful idea: separate **realistic vs oracle** F0.5 (blocking misses count as FN) |
| **IshaanVerma2204/aws_ml_challenge_2026** (25 Sep) | TF-IDF → TruncatedSVD 64-d → GPU cosine top-K; XGBoost + LR | "OOF 0.9486" on a *balanced pair sample* | [C] metric not the challenge metric. Predicted **2.29M pairs for 1.73M test S1** (≈1.3/S1 vs ~3.5 true), so severely under-recalling |
| amulyarai24-art/amazon-ml-challenge-2026 | Essay README: Fellegi-Sunter + LightGBM + Ditto + graph clustering | none | [C] |

### 2.3 Tier 3: initial research-set documents (as provided)
| Source | Useful content | Problems found |
|---|---|---|
| `amazon_ml_challenge_detailed_solution_methodology.md` | Sound generic plan: multiview normalization, multi-block union, provenance features, hard negatives, S1-grouped split, error-analysis loop | Hedges on country partition (now proven safe). Unaware of reverse retrieval and competition features |
| `amazon_ml_challenge_revised_competitive_solution.md` | Correctly adopts Ayan/jvikram ideas. Warns to verify exclusivity | Rates expected-F0.5 decoder as **Core**, but both source repos found it ties with a tuned threshold → demote. Assumes 16 GB / RTX 4050 laptop |
| `Amazon_ML_Challenge_2026_Revised_Solution.md` | Address decomposition, missingness features, candidate-context (margin) features | Same decoder over-rating. Otherwise consistent |
| `Amazon ML Hackathon Repositories.md` | Correct dataset stats (verified [M]) | Says "1-to-1" and recommends **Hungarian / Jonker-Volgenant**. That is wrong: the structure is many-to-one, and Hungarian would force S1s to at most one target and destroy recall. "Asymmetric loss" and a "Tier 2 >98.5% recall" are unsupported [C]. The hackathon window stated (9 AM – 9 PM) contradicts the official PDF |
| Pasted "Advanced Engineering Strategies" report (chat only, not saved to a file) | Useful framing of cascade, disjunctive blocking, density of distractors | Also claims "1-to-1 / Hungarian". "Heavily bias toward singletons" is misguided (94% have matches). Parquet "GB → tens of MB" is exaggerated. The GPU plan assumes GPU-heavy stages matter; evidence says CPU-bound. The Kaggle "0.105 baseline" is unverified |
| `deep-research-report.md` | 5 repo names | Timestamps look fabricated (all :00/:30). Parthrandive was actually created 24 Sep [M via API]. No technical content. Listed repos are tiny (46–259 KB) → low value |

### 2.4 Research batches ingested
| Batch | Date | Sources | New techniques |
|---|---|---|---|
| R1-a | 26 Sep | 4 local MDs, guideline PDF, pasted report | baseline inventory |
| R1-b | 26 Sep | GitHub sweep: 364 repos created 25–26 Sep (282 non-empty, 197 READMEs). 44 READMEs are verbatim copies of the problem statement. Deep read of the 8 repos in §2.1–2.2 | reverse retrieval, competition features, density simulation, LOCO protocol, collective stacking, learned cross-script dictionary, realistic-vs-oracle metric, relative-to-best rule |
| R2 | 26 Sep 21:45 | Re-sweep: 141 repos new or updated since 19:20 (131 non-empty, 97 READMEs). Public leaderboard snapshot from the user. Deep read: harshgitty58 (PROGRESS.md), mayankgoplani431-del (README + audit), Sid-techweb, Dhruv716-stack, Karthikatstuffmama (via the parallel session) | **validation→leaderboard gap evidence (§2.6)**, house-number guard, digit edit-type features, sibling expansion, hard per-S1 caps (S2 ≤ 5, S3 ≤ 6), dense e5 view for cross-script recall |

### 2.6 Leaderboard evidence (R2): validation does not transfer 1:1
Public leaderboard snapshot (user, 26 Sep ~21:40): ranks 4–30 all score **0.988–0.990**. The top reported elsewhere is 0.9906.

| Repo | Local validation | Public leaderboard | Gap | Note |
|---|---|---|---|---|
| harshgitty58 v1 → v2 | 0.9743 → 0.9842 | 0.967 → 0.969 | −0.007 → **−0.015** | +0.010 validation bought only +0.002 on the leaderboard. Probes: France is not the cause |
| mayankgoplani431 E02 | 0.9776 | 0.964 | −0.014 | blocking recall 95.4 → 97.5% transferred ~1.5× |
| mayankgoplani431 S3 | 0.9806 (↑) | 0.956 (↓) | −0.025 | + group/density-sensitive features: ~230k extra test pairs, 60–90% with a different house number than the S1 |
| mayankgoplani431 E10 (sibling features) | n/a | 0.933 | n/a | withdrawn |
| Sid-techweb v2/C4 | 0.98322 | 0.970441 | −0.013 | |
| Dhruv716-stack | 0.9102 | 0.894 | −0.016 | |

**Reading (inferred, consistent across 4 independent repos):**
- Test has about 2× as many distractors per S1 (5.75 vs 4.67 records per S1; at 3.46 true matches that is ~2.3 vs ~1.2 distractors).
- Features whose meaning depends on pool density (raw candidate counts, candidate-vs-candidate group features, "is this S1 the target's best", sibling support) make a train-fit model over-confident on test lookalike distractors. The typical case is the same name and street with a different house number.
- **Validation gains from such features can be leaderboard losses.**
- Recall gains (blocking) transfer; precision gains from context features may not.

**Consequences for our decisions:**
1. Judge A1 vs A2 on normal validation **and** density-stressed validation (`density.py`) **and** a label-free test check: predicted pairs per S1 on test vs validation (over-matching on test is a red flag).
2. House-number guard (features + decoder): separate noise (truncation, zero padding, suffix letter) from substitution (distractor).
3. Prioritize candidate recall (cross-script xs and dense dn views, parallel session).
4. Leaderboard slots are scarce (5/day) and the public board is a subset. Use them to test hypotheses (e.g. strict vs loose τ), not to hill-climb.

**Leaderboard teams' own repos (searched 26 Sep 22:10):** none found. All 26 team names on the user's snapshot (ranks 4–30) were checked against about 290 collected READMEs and GitHub repo search (name/description/README, created ≥15 Sep). Every hit was unrelated (printf libraries, "medallion" data pipelines, API spam). The only name-like match, MAVERICK-111/Amazon_ML_2026 (32 KB plan + HNSW scaffold, Apache-2.0, last push 15:29 IST), states no team name or score, so it is unconfirmed. Inferred: the 0.988–0.990 teams keep their code private. The best public evidence remains Ayan (validation 0.987, no leaderboard score) and the leaderboard-gap reports above.

### 2.5 Fair-play and licensing position
- **None** of the competitor repos has a license (default: all rights reserved), and the organisers screen for plagiarism. **We use them for ideas and measured facts only. No code is copied.** Everything we build is written from scratch.
- A repo named `TANISHQDAS/matching_results` (38 MB) appears to publish outputs. **Never download or use other teams' predictions.**
- Library and model licences to verify when adopted: LightGBM (MIT), XGBoost (Apache-2.0), RapidFuzz (MIT), sparse_dot_topn (Apache-2.0), polars (MIT), scikit-learn (BSD), potion-multilingual-128M (MIT), BGE-M3 (MIT), multilingual-MiniLM (Apache-2.0).
- Hand-written normalization lexicons (e.g., "R." → "rue", "SARL") are domain knowledge, not external data lookup. Data-learned maps (from train pairs) are safest.

---

## 3. Technique inventory (R1)

Worth: ★★★ must-have · ★★ test early · ★ test if time · ✗ skip. Cost is for the full pipeline on a 16-core / 19 GB machine.

### A. Data ingestion
| Technique | Solves | Source | Benefit | Cost | Worth |
|---|---|---|---|---|---|
| TSV → Parquet once, `QUOTE_NONE` | 3-min reloads, quote bug | Ayan, jvikram, akshatbakshi | fast iteration | 5 min | ★★★ |
| Integer row ids + string dictionaries | pandas object RAM blow-up | Ayan (`rid`) | 5–10× less RAM | low | ★★★ |
| Stage manifests / resume | crashes on long runs | jvikram | robustness | medium | ★ (simple file-exists caching instead) |

### B. Normalization (multi-view; never overwrite raw)
| Technique | Solves | Source | Benefit | Cost | Worth |
|---|---|---|---|---|---|
| NFKC, lowercase, fold accents, punctuation → space, glue single letters (L.L.C.→llc) | surface noise, France accents | all | large | low | ★★★ |
| Null-token removal (null, N/A), junk prefixes (`<<`, `--`), `M/s`, brackets | injected noise | Ayan, Tanmay | medium | low | ★★★ |
| Leetspeak repair (0→o, 1/l→i) in alpha tokens | `C0astal` | Ayan | small | low | ★★ |
| Domain/handle split (`coastaltungsten.com` → core) + word segmentation | domain-names | Ayan, jvikram (+0.4 pt recall) | medium | low-med | ★★ |
| Alias split (dba/aka/fka) | alias records | Ayan | small | low | ★ |
| Rule-based Indic → Latin transliteration (one table for 9 scripts, ISCII layout) | 7.3% cross-script pairs | Ayan, jvikram, Nikhil | India stratum | medium | ★★★ |
| **Learned native→Latin token dictionary** from aligned train pairs | transliteration spelling mismatch | Ayan (India name p10 sim 8 → 100) | large on India | medium | ★★ |
| Learned abbreviation maps (address and state codes) from train pairs | Rd/Road, TX/Texas, MH/Maharashtra | Ayan (addr p10 70 → 84–86) | medium | medium | ★★ |
| Legal-form / generic-token handling via **learned DF** ("core" tokens = DF ≤ 1%) | Corp/Ltd/SARL noise, works for France | Ayan | medium | low | ★★★ |
| House-number / number-token extraction (strip #, No, H.NO, leading zeros) | number noise | Ayan, Tanmay | feature quality | low | ★★★ |

### C. Blocking / candidate generation
| Technique | Solves | Source | Benefit | Cost | Worth |
|---|---|---|---|---|---|
| **Country partition** (label-agnostic) | search space | all; verified [M] | lossless, ~2× less work | none | ★★★ |
| Exact normalized-name key | trivial matches | docs | covers only 26% of pairs | low | ✗ as a view (subsumed by TF-IDF) |
| Inverted-index keys with DF caps and conjunctions | scalable exact-ish | jvikram, Akash, Tanmay | 0.894–0.965 recall | low-med | ✗ (dominated by C below) |
| MinHash LSH | typos | pasted report | none measured | med | ✗ (TF-IDF top-K is exact and better) |
| Postal-code keys | locality | docs | postcodes in <11% US / <2% India | low | ✗ |

### D. Retrieval (sparse top-K)
| Technique | Solves | Source | Benefit | Cost | Worth |
|---|---|---|---|---|---|
| Name char_wb 3-gram TF-IDF top-K | typos, reorder | Ayan, Nikhil, Raj | R@40 only 0.68 on full pool | 30–60 min | ★★ (as a union member) |
| Space-less name char 3-gram TF-IDF | domains, run-ons | Ayan | small union gain | same | ★ |
| Address word TF-IDF top-K | name-corrupted records | Ayan, Nikhil, Tanmay | R@40 0.886 | same | ★★ |
| **Combo view** [√.5·name ‖ √.5·addr] cosine | both fields agree | Ayan | R@10 0.960 | same | ★★★ |
| **Reverse view: target → S1 combo top-8** | same-name flooding; uses the many-to-one structure | Ayan | **R@1 0.972, R@8 0.9875; strongest single blocker** | ≈ same | ★★★ |
| Budget pruning (per-view K) | candidate volume | Ayan (K 40 → 5: −0.23 pt recall, 2.7× fewer) | fits our RAM | none | ★★★ |
| `sparse_dot_topn` multithreaded sparse matmul, chunked queries | speed | Ayan, Nikhil | enabling | — | ★★★ |
| TF-IDF → SVD dense + GPU cosine | GPU usage | Ishaan | lossy, unproven | med | ✗ |
| Dense multilingual ANN (BGE-M3, MiniLM) fallback for zero-candidate S1s | semantic misses | pasted report | tiny: union recall already 0.99 | high (GPU) | ✗ for blocking |

### E. Pairwise features
| Technique | Source | Worth |
|---|---|---|
| Exact TF-IDF cosines per view (name, cat, name-word, addr, combo) | Ayan | ★★★ |
| RapidFuzz: ratio, token_sort, token_set, partial, WRatio, Jaro-Winkler (name and address) | all | ★★★ |
| Token Jaccard, IDF-weighted Jaccard, IDF coverage each side, rarest shared token IDF | Ayan | ★★★ |
| Core-token Jaccard/equality (generic tokens removed via DF) | Ayan | ★★★ |
| Number features: Jaccard, first-number equal, shared count, missing flags, big-number Jaccard | Ayan, Tanmay | ★★★ |
| **House-number geometry** (numeric abs diff, edit distance, relative gap) | Ayan v2, jvikram `numdist` | ★★ (targets sibling FP; unmeasured) |
| `tokalign`: rarest unmatched token each side | jvikram | ★★ (unmeasured) |
| Acronym/initials, truncation-prefix, domain-prefix coverage, alias-part Jaccard | Ayan | ★★ |
| Phonetic-key Jaccard | Ayan | ★ |
| Script flags, target source (S2/S3), empty-address flags, lengths | Ayan, Tanmay | ★★★ |
| Name frequency among S1 (`q_nm_freq_s1`): ambiguity | Ayan | ★★★ |
| Retrieval provenance: per-view score and rank (999 if absent) | Ayan, docs | ★★★ |

### F. Hard-negative mining
| Technique | Source | Worth |
|---|---|---|
| **Train on the blocker's own candidates** (negatives = retrieved non-matches) | all strong repos | ★★★ (natural hard negatives, no special sampling) |
| Density-matched training (drop ~19% of S1s so their targets become distractors) | Ayan (+0.0014 under dense eval) | ★★ |
| Ditto-style augmentation (swap/delete) | Ditto via pasted report | ✗ (GBDT on real candidates suffices) |
| OOF-mined false positives upweighted (duplicate row = weight 2) | origin/hard-negative-mining (Yashwin69); `hard_negative_lab.py` (A2 feat2), `hard_negative_s2.py` (s2lab / dense_mix) | ? (tooling smoke-tested; not measured on real folds yet, E-X10) |

### G. ML models
| Technique | Source | Evidence | Worth |
|---|---|---|---|
| Rules / Fellegi-Sunter | docs, amulyarai | 0.767 best rule [R] | ✗ (sanity only) |
| **LightGBM on pair features** | all | 0.9846 [R] | ★★★ |
| XGBoost (GPU hist) | jvikram, Ishaan | not compared | ★ (swap-in if LightGBM training is slow) |
| k-fold cross-fit ensemble | Ayan | +0.0004 [R] | ★ |
| Cross-encoder reranker (Ditto/MiniLM) on ambiguous band | pasted report, docs | none measured | ★ high-risk (§4.4) |
| LLM matcher | — | too slow, no evidence | ✗ |

### H. Calibration
| Technique | Source | Evidence | Worth |
|---|---|---|---|
| None (LightGBM is well calibrated at train density) | Ayan | reliability table [R] | default |
| Isotonic on held-out | jvikram, Tanmay | needed only for the expected-F decoder | ★ |

### I. Collective / context features
| Technique | Source | Evidence | Worth |
|---|---|---|---|
| **Competition (many-to-one) features:** rank/gap/margin of q among all S1 competing for target t; rank/gap of t within q's list | Ayan | **top importance** [R] | ★★★ |
| Per-entity relative features (score / best, rank, z-score, count) | Tanmay, Revised doc | part of above | ★★★ |
| **Stage-2 stacking on OOF stage-1 p + sibling support** (target–target similarity among q's candidates, near-duplicate co-candidates) | Ayan | **+0.0021 (13 SE)** [R] | ★★ (largest measured post-baseline gain; costs a 2nd feature/train pass) |
| Connected-components clustering | Splink, amulyarai | address groups span several S1s 12–21% of the time [jvikram] | ✗ |

### J–M. Decision, F0.5, singletons, exclusivity
| Technique | Source | Evidence | Worth |
|---|---|---|---|
| **Global threshold tuned on macro F0.5** (τ ≈ 0.7–0.8) | all | best or tied [R] | ★★★ |
| **Per-target argmax exclusivity** (each target kept only for its best S1) | Ayan, Raj, Akash | ≈0 on top of competition features [R]; +0.0005–0.0007 [S] | ★★★ (free, safe) |
| Relative rule `p ≥ τ and p ≥ best − δ`, cap per S1 | Tanmay | [S] | ★ |
| Source-specific τ (S2 vs S3) | Akash | +0.0003 [S] | ★ |
| Expected-F0.5 set decoder (Poisson-binomial DP) | jvikram, Ayan, docs | ties/loses vs threshold [R] | ★ (cheap to try, low expectation) |
| Match-exists (π) model for singletons | Ayan | no gain [R] | ✗ |
| Hungarian / Jonker-Volgenant 1-to-1 | Hackathon Repositories.md, pasted report | **wrong structure** | ✗ |
| Asymmetric FP-weighted loss | Hackathon Repositories.md | none | ✗ (threshold does this) |

### N. Embedding-based methods
| Technique | Source | Evidence | Worth |
|---|---|---|---|
| Static multilingual embeddings (potion-multilingual-128M, MIT, CPU) as cosine features | Ayan | computed; gain not isolated | ★ |
| BGE-M3 / multilingual-MiniLM dense features for France | pasted report | none | ★ (only if the France proxy shows a gap that lexical features can't close) |

### O. Other
| Technique | Source | Evidence | Worth |
|---|---|---|---|
| **Realistic vs oracle F0.5** reporting | Raj, Ayan | protocol | ★★★ |
| LOCO (US↔India) as France proxy | Ayan, Raj | gap 0.01–0.04 [R] | ★★ |
| Conservative τ for unseen country labels | Ayan (OOD optimum τ shifts up) | [R] | ★★ |
| Self-training on unlabeled France test (pseudo-labels p ≥ 0.99 and exclusive-best) | Ayan | +0.002 / −0.002 mixed [R] | ★ (transductive; allowed since label-free, but risky) |
| Learning curve: more training folds | Ayan (+0.0008 per doubling) | [R] | ★★ (final model only) |

---

## 4. Candidate architectures

### 4.1 Minimum viable baseline (MVB): the first leaderboard submission
Country partition → normalization v0 (NFKC, fold accents, punctuation, generic cleanup; no transliteration) → TF-IDF views **combo top-10 + reverse top-8 + addr top-5 + name top-5** → ~30 core features (cosines, RapidFuzz, token/number, provenance, empty flags, source) → LightGBM trained on 1–2 folds → τ sweep + per-target argmax.
Expected: well above 0.95 (TF-IDF + reverse gives ~0.99 recall [R]; LightGBM with basic features reached 0.967–0.976 [S] with weaker blocking).

### 4.2 Strong classical pipeline (target for the final submission)
MVB + Indic transliteration and learned native dictionary + learned abbreviation maps + full feature set (E-groups) + **competition features** + density-matched validation and threshold + training on more folds. Expected ≈ 0.984–0.985 validation [R, Ayan cv_v1].

### 4.3 Advanced pipeline
Strong + OOF stage-1 → collective/sibling features → stage-2 LightGBM + house-number geometry features + conservative τ for unseen countries. Expected ≈ 0.987 validation [R]. Costs roughly 2× feature/training time.

### 4.4 Experimental / high-risk
- Cross-encoder reranker (multilingual MiniLM, Apache-2.0) on the ambiguous band only (0.3 < p < 0.9, a small fraction of pairs), as a stage-2 feature. **This is the only place a GPU could matter.** No evidence yet on this data.
- Dense multilingual embedding features for France.
- Self-training on France test.
- Expected-F0.5 decoder.

### 4.5 Components that must be validated before joining the final system
| Component | Why unproven for us |
|---|---|
| Learned native dictionary / abbreviation maps | Leak risk if learned on validation-fold pairs. Must learn on train folds only |
| Stage-2 stacking | Ayan's gain partly used a feature that "may not transfer". Re-verify |
| House-number geometry, tokalign | Unmeasured anywhere |
| Density-matched τ | Hypothesis. Check against the first LB submission |
| Everything France-specific | No labels. Only a LOCO proxy |
| Any neural component | No evidence on this data |

---

## 5. Experiment matrix

Principle: one change per experiment; everything else frozen. "Keep if" uses the P-main fold with bootstrap SE (§6). A change is real if Δ > 2 SE (≈ 0.0003 at 220k S1). Cost is estimated for one 16-core machine.

| ID | Hypothesis | Change | Fixed | Expected effect | Metrics | Cost | Machine | Keep if |
|---|---|---|---|---|---|---|---|---|
| **E0** | We need a trustworthy local score | Parquet + int ids; exact metric (unit test = 0.714 example); fold = S1 id mod 10; reporting of all §6 metrics | — | enables everything | metric test passes; load < 30 s | 1–2 h | any | — (infra) |
| **E1** | TF-IDF combo + reverse retrieval gives ≥0.985 pair recall at ≤50 cands/S1 on the full pool | Normalization v0 + 4 views + union; K sweep | E0 | recall 0.985–0.99 | pair recall, entity recall, oracle F0.5, cands/S1 (mean/P95), time, RAM | 2–4 h (train+test) | highest-RAM box | recall ≥ 0.98 at ≤ 60 cands/S1 |
| **E2** | LightGBM on core features beats rules by >0.2 | ~30 feats, train folds 1–2, τ sweep + argmax exclusivity | E1 candidates | 0.96–0.98 | macro F0.5, P, R, singleton acc, FP/FN counts, realistic vs oracle | 1–2 h | same | ≥ 0.95 → **submit (LB #1)** |
| **E3** | Many-to-one competition features are the top signal | + trank/tgap/tmargin, qrank/qgap, q_ncand, t_nq, name frequency | E2 | +0.005–0.02 | ΔF0.5, feature importance | 30–60 min | same | Δ > 2 SE |
| **E4** | Cross-script bridge closes the India gap | + Indic→Latin transliteration; then + learned native dictionary (train folds only) | E3 | India stratum up, blocking recall on Indic targets up | ΔF0.5 overall and India/brahmic strata, Δrecall | 2–3 h (re-block India) | box B | Δ > 2 SE overall or India-stratum Δ > 0.005 |
| **E5** | Learned abbreviation maps + full token/number features help | + maps, IDF/core/number/acronym/domain features | E4 | +0.002–0.005 | ΔF0.5, US/India strata | 1–2 h | box B | Δ > 2 SE |
| **E6** | Train-density τ is optimistic on test | P-dense simulation (drop 19% of S1 → distractors), re-tune τ; optionally train at dense | E5 | τ 0.7 → ~0.8 | F0.5 vs τ curve, dense vs normal | 1–2 h | box C | P-dense τ tracks LB better (check with LB #2) |
| **E7** | France-like shift needs a higher τ / generic features | LOCO US↔India with the E5 system; τ for unseen labels | E5 | quantify gap (0.01–0.04) | LOCO F0.5, OOD τ | 1–2 h | box C | use its τ offset for France if LOCO optimum differs |
| **E8** | Collective stacking gives +0.002 | 3-fold OOF stage-1 → sibling/support features → stage-2 | E5 | +0.001–0.002 | ΔF0.5, pair P | 3–5 h | box D (after E5) | Δ > 2 SE and time allows |
| **E9** | Sibling FPs fall with house-number geometry | + numdist/tokalign features | E5 | pair P up | ΔF0.5, FP on sibling detector | 1 h | box B | Δ > 2 SE |
| **E10** | Decoder variants add little | expected-F DP, source-specific τ, relative-to-best δ | E5 scores | ±0.0005 | ΔF0.5 | 30 min | any | Δ > 2 SE (else keep simple τ) |
| **E11** | Neural ambiguous-band reranker adds signal | fine-tune multilingual MiniLM cross-encoder on hard pairs; add its score as a stage-2 feature | E8 | unknown | ΔF0.5 on ambiguous band and overall | 4–8 h | **RTX 5060** | Δ > 2 SE; only if E1–E8 are done early |
| **E12** | More training data helps the final model | train on folds 1–9 with fixed rounds | final config | +0.0005–0.001 | fold-0 check | 1–2 h | highest-RAM box | final submission only |

Dependency order: E0 → E1 → E2 (LB #1) → E3 → {E4, E5 in parallel} → E6 & E7 (need E5 scores) → E8 / E9 → E10 → E12 → final. E11 runs in parallel only if a free GPU machine exists and E5 is done.

---

## 6. Evaluation methodology

### 6.1 Splits (S1-grouped; avoid fooling ourselves)
- `fold = int(S1 id digits) % 10`. S1 ids are random [R: Spearman ≈ 0 with row order], and every target belongs to exactly one S1, so no true pair spans folds.
- **P-main:** train folds {1,2}, early-stopping/τ fold {3}, report fold {0} (~220k S1). **Choose τ on fold 3, report on fold 0**. Never choose and report on the same fold.
- **Full target pool always:** validation S1 query against *all* train S2/S3. Subsampled pools overstate recall by up to 5 points [R, jvikram].
- **P-dense:** density-shifted validation (§6.3). **P-loco:** US→India and India→US as the France proxy.
- Label-free corpus statistics (IDF, name frequency, competition ranks) may use all folds: the test gets identical treatment. **Learned maps may only use training-fold pairs.**
- Fold 9: untouched audit, used once before the final submission.

### 6.2 Metrics per stage
| Stage | Metrics |
|---|---|
| Normalization | true-pair name/addr token-set similarity p10/p50 per country/script (before vs after); runtime |
| Blocking | pair recall, entity full recall, **oracle macro F0.5**, cands/S1 (mean, P95, max), reduction ratio, per-view R@K, recall by country × source × script, runtime, peak RAM |
| Matching | pair AUC/AP, pair P/R at τ, feature importance, train time, peak RAM |
| Decision | **realistic macro F0.5** (blocking misses count as FN), macro P, macro R, **singleton accuracy**, F0.5 by n_true bucket (0, 1, 2–3, ≥4), by country, by source, **FP count and FP on singletons**, FN-in-candidates vs blocking misses |
| End-to-end | full test runtime, peak RAM, VRAM (if any), validator PASS, predicted pairs/S1 on test (sanity: should be ~3–4, not ~1.3 as in Ishaan's submission) |
| Leaderboard | public score vs local P-main and P-dense estimates; the gap decides which protocol we trust |

### 6.3 Test/validation mismatch controls
- **Density:** test has 23% more targets per S1. Simulate by dropping a random ~19% of train S1s from the graph so their targets become unlabeled distractors, recompute competition features, and evaluate [R, Ayan].
- **Unseen country:** LOCO gap and OOD τ shift. A leaderboard-predictive estimate ≈ 0.85·F(P-dense) + 0.15·F(LOCO-like).
- **Bootstrap SE** over S1 (200 resamples) is reported with every headline number.

### 6.4 Error-analysis loop (after every accepted change)
Classify errors with label-free detectors: blocking miss / empty target address / shared S1 name / not best S1 for target / sibling (same street, nearby number) / cross-script / truncation / domain / alias. Change only the component that owns the biggest bucket.

---

## 7. GPU / machine allocation plan

The honest finding: **no measured-strong component in this competition needs a GPU.** Allocate by CPU and RAM, and shard by country (India is the largest unit, France the smallest).

| Machine | Role (proposed) | Why |
|---|---|---|
| **Box A: highest RAM** (unknown which) | E0–E2 main line: blocking train + test, first submission, final E12 training | Blocking and feature extraction are RAM-bound |
| **Box B** | India-specific work: E4 transliteration and dictionary, E5 features, E9; re-block India | India is the largest partition and has the cross-script problem |
| **Box C** | Validation science: E6 P-dense, E7 LOCO, E10 decoder variants | Needs only cached features and scores; low RAM |
| **Box D: RTX 5060** (most VRAM) | E8 stacking (CPU), then E11 cross-encoder (GPU) only if time remains; optional embedding features | The only GPU-relevant experiments |
| This machine (3050, 16 threads, 19 GB) | Can be Box A if the others have less RAM | 12 GB free now; close other apps |

Sharing: data and artifacts as Parquet on a shared drive or synced folder. Every box uses the same `fold` definition and the same frozen normalization version.

**Needed from you:** RAM and CPU core count for the other three machines, and whether they share a network drive.

---

## 8. Risks and unknowns
| Risk | Impact | Mitigation |
|---|---|---|
| **Time:** ~28 h left. Ayan's full pipeline took >1 day on 32 vCPU/128 GB | may not reach §4.3 | MVB first; every later step is incremental and submittable |
| **Unused submissions expire daily** (5 today, ending 23:59 IST) | lose leaderboard calibration points | aim for LB #1 as early as possible |
| RAM: full test ≈ 1.73M × 45 ≈ 78M pairs | OOM | chunk by country and S1 shard; keep only scores; float32 features |
| France unmeasurable | 15% of the score | LOCO proxy, generic learned-DF features, conservative τ offset for unseen labels |
| Density shift | −0.0015 and τ drift | P-dense τ; check against LB |
| Validation leak via learned maps / transductive features | optimistic local score | learn maps on train folds only; report realistic metric |
| Library installs (polars, lightgbm, rapidfuzz, sparse_dot_topn) | setup delay | install first thing; sparse_dot_topn has a scipy fallback (slower) |
| Plagiarism screening | disqualification | write our own code; cite ideas in the methodology doc |
| New research contradicting R1 | plan churn | §10 delta log; re-rank only on [R]-grade evidence |

Open questions: other machines' specs; whether teammates will run code in parallel; team name for packaging.

---

## 9. Recommended first experiment

**E0 + E1: evaluation harness plus TF-IDF blocking with the reverse view, on the full train pool, then straight into E2 for leaderboard submission #1.**

Why this first:
1. Blocking sets the recall ceiling, and the evidence is decisive: key-based blocking plateaus at 0.89–0.96 [R/S], while TF-IDF combo + reverse reaches 0.99 [R]. Getting this right once avoids redoing everything downstream.
2. It builds the harness (exact metric, folds, recall and oracle numbers) that every later experiment needs.
3. E1 → E2 is the shortest path to a validated leaderboard submission, and the first public score tells us how far local validation is from test.

Concrete E1 plan: install packages → Parquet with int ids → normalization v0 → per country: fit TF-IDF (name char_wb 3-gram; address words; combo) on a ≤3M sample → S1→T top-K per view and T→S1 top-8 via chunked sparse matmul → union → report the §6.2 blocking metrics on folds 0–3 → also run on test and write `candidate_pairs.tsv`.

Decision needed from you: approve E0 + E1 (or pick another), and send the other machines' RAM/CPU.

---

## 10. Revision log
| Rev | Date | Change |
|---|---|---|
| R1 | 2026-09-26 19:30 IST | Initial synthesis: 4 local docs, pasted report, guideline PDF, own data verification, GitHub sweep of 364 challenge-window repos (8 read in depth) |
| R2 | 2026-09-26 21:50 IST | Leaderboard snapshot + re-sweep (141 new/updated repos). New §2.6: validation→leaderboard gap of 0.013–0.025 across 4 repos, attributed to test distractor density. A1/A2 selection now also requires density-stressed validation and a test over-matching check; house-number guard added |
| R3 | 2026-09-27 11:30 IST | Re-sweep, Siva402-ai/amazon-ml-challenge-2026 read in depth. Their facts: v2/C4 LB 0.970441 at local DEV 0.98322 (gap −0.013, like ours −0.010). A sibling/cluster-agreement feature run (E10) scored LB 0.933, so they now require features that are direct S1↔record evidence and invariant to duplicating other records (0 decision flips under k=1..8 duplication). Their US↔India transfer proxy puts an unseen country at ~0.93–0.945 (ours: E-G5 0.922), with one self-training round at p≥0.98/≤0.02 worth +0.0027 averaged over both directions, and a second round hurting (confirmation bias). France-only v4: French-locale normalisation (region names ↔ department codes, legal/filler words such as Compagnie/Cie) plus self-training at τ 0.9; LB not reported. v3 'roles' features (premise/unit/floor numbers) gave +0.001 locally, LB untested. Ideas only, no code copied (unlicensed repo). |
| R4 | 2026-09-27 11:55 IST | LB-gap re-analysis (EXPERIMENTS E-G8). harshgitty58 blank probes re-read: the US drop 0.391 exceeds US's 38.3% share of test S1, so the public set is not population-weighted (≈ US 0.43 / India 0.43 / France 0.146), and per-country F is only identified up to w_c (France blank → F_FR ± 0.05). Differential country-isolated uploads are exact. New public gap point: Abhigyanv23 holdout 0.9773 → LB 0.9683 (−0.009); seven pipelines lose 0.007–0.016. Test failure modes found label-free: planted affiliate look-alikes kept (US/India +group/+holdings/+ventures…, 0% true in train) and France same-street/different-house-number look-alikes kept (France true copies change the number 0.4% vs US 6.9%). Idea sources: harshgitty58 brainstorm #3 (distractors substitute house numbers) and Siva402 (features must be direct S1↔record evidence). No code copied. |
| R5 | 2026-09-27 16:10 IST | E-G9 guard audit. Train-learned affiliate veto is clean in every labeled world (0 true removed; +0.0035 in the unseen-country proxy) and ~90% planted on test by the house-number signature; expected LB +0.0005–0.0007. Both label-free France rules rejected: lfveto (planted and noise tokens overlap; France kept +groupe pairs keep the house number 95%, i.e. true copies) and frhn (its evidence was selection bias; true-copy sample trips it 2.7% vs 1.2% overall). New discriminator for any look-alike rule: planted look-alikes shift the house number, true copies keep it. OpenCode's entity abstention and sibling-conflict features are null. |
