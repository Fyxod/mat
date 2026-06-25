# Semantic rescore of legacy Phase 1A/1B

This pass re-ranks legacy internal-surrogate candidates by final-edit-aware semantic scoring.
If CLIP is unavailable, rows are intentionally labeled metric-only rather than strong.

## phase1a

- Rows in source CSV: 108
- Rows semantically scored: 108
- Rows missing images: 0
- Decision counts: {'metric_only_candidate': 106, 'weak_candidate': 2}

- make_the_person_smile_slightly / vae_conditioning / strong: metric_only_candidate final=0.2525, semantic_drop=0.0000
- add_a_small_beard / vae_conditioning / strong: metric_only_candidate final=0.2438, semantic_drop=0.0000
- make_the_person_smile_slightly / vae_conditioning / strong: metric_only_candidate final=0.2413, semantic_drop=0.0000
- add_a_small_beard / unet_prediction / strong: metric_only_candidate final=0.2370, semantic_drop=0.0000
- add_black_sunglasses / edit_direction / strong: metric_only_candidate final=0.2302, semantic_drop=0.0000

## phase1b

- Rows in source CSV: 48
- Rows semantically scored: 48
- Rows missing images: 0
- Decision counts: {'metric_only_candidate': 48}

- add_a_small_beard / unet_prediction / strong: metric_only_candidate final=0.2908, semantic_drop=0.0000
- add_a_small_beard / unet_prediction / strong: metric_only_candidate final=0.2901, semantic_drop=0.0000
- make_the_person_smile_slightly / vae_conditioning / strong: metric_only_candidate final=0.2796, semantic_drop=0.0000
- make_the_person_smile_slightly / vae_conditioning / strong: metric_only_candidate final=0.2348, semantic_drop=0.0000
- add_a_small_beard / vae_conditioning / strong: metric_only_candidate final=0.2172, semantic_drop=0.0000

