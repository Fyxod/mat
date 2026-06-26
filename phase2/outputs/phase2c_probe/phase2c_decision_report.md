# Phase 2C targeted headphone failure probe decision report

Phase 2C did not find a visible headphone edit failure. The best candidates only weaken or shift the headphones. Do not spend more A6000 time on the same InstructPix2Pix/headphones setup.

## Scope

- Prompt: add headphones
- Perturbation type: geometric coordinate warps only
- Parts: amplification, headphone-region ablations, semantic-heavy CEM
- A row is not a true success unless the clean edit clearly adds headphones and the perturbed edit does not clearly add headphones.

## Counts

- Total candidates: 895
- Visible failure candidates: 0
- Strong semantic candidates: 0
- Weak semantic candidates: 314
- Metric-only candidates: 529
- Rejected for input damage: 52
- Part counts: {'amplification': 315, 'region_ablations': 60, 'semantic_heavy_cem': 520}

## Best semantic rows

- amplification / region_local_tps / source_regions / medium amp_01_cand_034_g03_m02_x1p0_medium: label=weak_candidate, score=0.2905, semantic_drop=0.0320, perturbed_margin=-0.0041, input_ssim=0.9757, max_disp=1.43
- amplification / region_local_tps / source_regions / strong amp_01_cand_034_g03_m02_x1p0_strong: label=weak_candidate, score=0.2905, semantic_drop=0.0320, perturbed_margin=-0.0041, input_ssim=0.9757, max_disp=1.43
- amplification / region_local_tps / source_regions / existence_probe amp_01_cand_034_g03_m02_x1p0_existence_probe: label=weak_candidate, score=0.2905, semantic_drop=0.0320, perturbed_margin=-0.0041, input_ssim=0.9757, max_disp=1.43
- semantic_heavy_cem / region_local_tps / headphones_full / strong cand_021_g02_m05: label=weak_candidate, score=0.2816, semantic_drop=0.0311, perturbed_margin=-0.0033, input_ssim=0.9805, max_disp=1.53
- semantic_heavy_cem / region_local_tps / headphones_sides_only / strong cand_064_g04_m16: label=weak_candidate, score=0.2792, semantic_drop=0.0309, perturbed_margin=-0.0030, input_ssim=0.9700, max_disp=1.87
- semantic_heavy_cem / region_local_tps / headphones_full / strong cand_007_g01_m07: label=weak_candidate, score=0.2785, semantic_drop=0.0306, perturbed_margin=-0.0028, input_ssim=0.9701, max_disp=1.71
- semantic_heavy_cem / region_local_tps / headphones_full / existence_probe cand_041_g03_m09: label=weak_candidate, score=0.2843, semantic_drop=0.0304, perturbed_margin=-0.0026, input_ssim=0.9505, max_disp=2.07
- semantic_heavy_cem / region_local_tps / headphones_full / strong cand_030_g02_m14: label=weak_candidate, score=0.2760, semantic_drop=0.0302, perturbed_margin=-0.0023, input_ssim=0.9712, max_disp=1.80
- semantic_heavy_cem / region_local_tps / headphones_full / strong cand_055_g04_m07: label=weak_candidate, score=0.2784, semantic_drop=0.0302, perturbed_margin=-0.0023, input_ssim=0.9688, max_disp=1.45
- semantic_heavy_cem / region_local_tps / headphones_full / strong cand_056_g04_m08: label=weak_candidate, score=0.2751, semantic_drop=0.0300, perturbed_margin=-0.0021, input_ssim=0.9698, max_disp=1.85
- amplification / region_local_tps / source_regions / medium amp_02_cand_025_g02_m09_x1p0_medium: label=weak_candidate, score=0.2729, semantic_drop=0.0296, perturbed_margin=-0.0017, input_ssim=0.9684, max_disp=1.56
- amplification / region_local_tps / source_regions / strong amp_02_cand_025_g02_m09_x1p0_strong: label=weak_candidate, score=0.2729, semantic_drop=0.0296, perturbed_margin=-0.0017, input_ssim=0.9684, max_disp=1.56

## Visual inspection rule

If the perturbed edit still clearly has headphones, call it weak or metric-only even if semantic_drop improved.
