# PHASE1C decision report

- Completed starts: 96
- Strong semantic candidates: 0
- Weak semantic candidates: 27
- Metric-only candidates: 68
- Rejected for input damage: 1
- Rejected because clean edit was weak: 0

Rows are ranked by `final_attack_score` for semantic decisions. The old `attack_score` is retained only as an output-disruption diagnostic.

## weak_candidate

- add_headphones / multi_timestep_edit_direction / medium start 3 iter 180: final=0.2412, semantic_drop=0.0276, input_ssim=0.9474, old_attack=0.1861
- add_headphones / multi_timestep_hybrid / medium start 0 iter 45: final=0.2367, semantic_drop=0.0298, input_ssim=0.9495, old_attack=0.1771
- add_headphones / multi_timestep_hybrid / medium start 1 iter 135: final=0.2352, semantic_drop=0.0327, input_ssim=0.9727, old_attack=0.1698
- add_headphones / multi_timestep_edit_direction / medium start 0 iter 135: final=0.2350, semantic_drop=0.0299, input_ssim=0.9588, old_attack=0.1753
- add_headphones / multi_timestep_hybrid / medium start 3 iter 90: final=0.2327, semantic_drop=0.0316, input_ssim=0.9678, old_attack=0.1695
- add_headphones / multi_timestep_unet_prediction / strong start 2 iter 135: final=0.2271, semantic_drop=0.0151, input_ssim=0.9368, old_attack=0.1969
- add_headphones / multi_timestep_unet_prediction / medium start 0 iter 90: final=0.2198, semantic_drop=0.0271, input_ssim=0.9596, old_attack=0.1656
- add_headphones / multi_timestep_edit_direction / strong start 3 iter 45: final=0.2179, semantic_drop=0.0264, input_ssim=0.9579, old_attack=0.1652

## metric_only_candidate

- add_a_small_beard / multi_timestep_edit_direction / strong start 0 iter 90: final=0.2911, semantic_drop=0.0012, input_ssim=0.9004, old_attack=0.2887
- add_headphones / multi_timestep_edit_direction / strong start 1 iter 135: final=0.2863, semantic_drop=-0.0070, input_ssim=0.9010, old_attack=0.3002
- add_a_small_beard / multi_timestep_hybrid / strong start 0 iter 135: final=0.2826, semantic_drop=0.0020, input_ssim=0.9025, old_attack=0.2786
- add_a_small_beard / multi_timestep_edit_direction / strong start 3 iter 135: final=0.2706, semantic_drop=0.0003, input_ssim=0.9010, old_attack=0.2700
- add_a_small_beard / multi_timestep_edit_direction / strong start 1 iter 180: final=0.2556, semantic_drop=0.0016, input_ssim=0.9121, old_attack=0.2525
- add_round_glasses / multi_timestep_unet_prediction / strong start 1 iter 135: final=0.2492, semantic_drop=0.0035, input_ssim=0.9166, old_attack=0.2423
- add_a_small_beard / multi_timestep_hybrid / strong start 2 iter 45: final=0.2401, semantic_drop=0.0012, input_ssim=0.9200, old_attack=0.2376
- add_black_sunglasses / multi_timestep_edit_direction / strong start 0 iter 45: final=0.2308, semantic_drop=-0.0077, input_ssim=0.9140, old_attack=0.2462

## reject_input_damage

- add_round_glasses / multi_timestep_edit_direction / medium start 3 iter 90: final=-0.1303, semantic_drop=0.0035, input_ssim=0.9174, old_attack=0.0437

