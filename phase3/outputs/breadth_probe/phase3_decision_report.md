# Phase 3 breadth probe decision report

- Total candidates: 816
- Visible failure candidates: 0
- Strong semantic candidates: 0
- Weak semantic candidates: 0
- Metric-only candidates: 773
- Rejected for input damage: 43

Ranking prioritizes final-edit semantic weakening over generic output pixel difference. If the clean and perturbed final edits visually still show the requested edit, treat the row as weak or metric-only.

## metric_only_candidate

- face_002 / add a hoodie / igs=1.0 / region_local_tps / strong / cand_012_g02_m04: score=0.5696, semantic_drop=0.0025, clean_margin=0.0475, perturbed_margin=0.0450, input_ssim=0.9280
- face_002 / add a hoodie / igs=1.0 / combined_tps_dct / strong / cand_013_g02_m05: score=0.5364, semantic_drop=0.0025, clean_margin=0.0475, perturbed_margin=0.0450, input_ssim=0.9241
- face_002 / add a hoodie / igs=1.0 / region_local_tps / strong / cand_003_g01_m03: score=0.4555, semantic_drop=0.0026, clean_margin=0.0475, perturbed_margin=0.0449, input_ssim=0.9385
- face_002 / add a hoodie / igs=1.0 / region_local_tps / strong / cand_016_g02_m08: score=0.4473, semantic_drop=0.0033, clean_margin=0.0475, perturbed_margin=0.0442, input_ssim=0.9471
- face_002 / add a hoodie / igs=1.0 / combined_tps_dct / strong / cand_014_g02_m06: score=0.4410, semantic_drop=0.0032, clean_margin=0.0475, perturbed_margin=0.0443, input_ssim=0.9216
- face_002 / add a hoodie / igs=1.0 / combined_all / strong / cand_008_g01_m08: score=0.4085, semantic_drop=0.0020, clean_margin=0.0475, perturbed_margin=0.0455, input_ssim=0.9449
- face_008 / add headphones / igs=1.0 / region_local_mesh / strong / cand_010_g02_m02: score=0.3967, semantic_drop=0.0013, clean_margin=0.0454, perturbed_margin=0.0441, input_ssim=0.9254
- face_008 / add headphones / igs=1.0 / region_local_dct / strong / cand_011_g02_m03: score=0.3889, semantic_drop=0.0017, clean_margin=0.0454, perturbed_margin=0.0437, input_ssim=0.9098
- face_008 / add headphones / igs=1.0 / combined_tps_dct / strong / cand_014_g02_m06: score=0.3862, semantic_drop=0.0009, clean_margin=0.0454, perturbed_margin=0.0446, input_ssim=0.9096
- face_002 / add a hoodie / igs=1.0 / region_local_tps / strong / cand_011_g02_m03: score=0.3850, semantic_drop=0.0018, clean_margin=0.0475, perturbed_margin=0.0457, input_ssim=0.9250
- face_004 / add a hoodie / igs=1.0 / combined_all / strong / cand_007_g01_m07: score=0.3795, semantic_drop=-0.0006, clean_margin=0.0409, perturbed_margin=0.0416, input_ssim=0.9802
- face_002 / add headphones / igs=1.0 / region_local_tps / strong / cand_012_g02_m04: score=0.3795, semantic_drop=0.0060, clean_margin=0.0512, perturbed_margin=0.0452, input_ssim=0.9064

## reject_input_damage

- face_008 / add headphones / igs=1.0 / combined_tps_dct / strong / cand_001_g01_m01: score=0.3600, semantic_drop=0.0014, clean_margin=0.0454, perturbed_margin=0.0441, input_ssim=0.8990
- face_002 / add headphones / igs=1.0 / combined_all / strong / cand_002_g01_m02: score=0.3108, semantic_drop=0.0032, clean_margin=0.0512, perturbed_margin=0.0480, input_ssim=0.8981
- face_002 / add headphones / igs=1.5 / combined_all / strong / cand_005_g01_m05: score=0.2809, semantic_drop=0.0014, clean_margin=0.0468, perturbed_margin=0.0455, input_ssim=0.8973
- face_003 / add headphones / igs=1.0 / combined_tps_dct / strong / cand_015_g02_m07: score=0.2737, semantic_drop=0.0007, clean_margin=0.0458, perturbed_margin=0.0451, input_ssim=0.8948
- face_003 / add headphones / igs=1.0 / combined_tps_dct / strong / cand_012_g02_m04: score=0.2674, semantic_drop=0.0027, clean_margin=0.0458, perturbed_margin=0.0431, input_ssim=0.8970
- face_008 / add headphones / igs=1.0 / region_local_tps / medium / cand_008_g01_m08: score=0.2553, semantic_drop=0.0008, clean_margin=0.0454, perturbed_margin=0.0446, input_ssim=0.9386
- face_002 / add headphones / igs=1.5 / region_local_tps / medium / cand_016_g02_m08: score=0.2306, semantic_drop=-0.0008, clean_margin=0.0468, perturbed_margin=0.0476, input_ssim=0.9396
- face_003 / add headphones / igs=1.0 / combined_tps_dct / strong / cand_008_g01_m08: score=0.2245, semantic_drop=0.0013, clean_margin=0.0458, perturbed_margin=0.0444, input_ssim=0.8951
- face_002 / add headphones / igs=1.5 / region_local_tps / medium / cand_014_g02_m06: score=0.1811, semantic_drop=-0.0005, clean_margin=0.0468, perturbed_margin=0.0473, input_ssim=0.9381
- face_002 / add headphones / igs=1.0 / region_local_tps / medium / cand_008_g01_m08: score=0.1807, semantic_drop=0.0007, clean_margin=0.0512, perturbed_margin=0.0505, input_ssim=0.9346
- face_008 / add headphones / igs=1.0 / combined_tps_dct / strong / cand_005_g01_m05: score=0.1624, semantic_drop=0.0016, clean_margin=0.0454, perturbed_margin=0.0438, input_ssim=0.8881
- face_003 / add headphones / igs=1.0 / region_local_tps / medium / cand_003_g01_m03: score=0.1545, semantic_drop=-0.0010, clean_margin=0.0458, perturbed_margin=0.0467, input_ssim=0.9345

## Recommendation

No promising candidates were found in the tested breadth probe. Consider a larger image set or another edit model rather than deepening the same setup.
