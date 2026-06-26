# Phase 3 clean discovery report

- Detected face folders: 8
- Clean discovery evaluations: 198
- Selected for first breadth probe: 24
- Rejected clean cases: 19
- Skipped incompatible image/prompt/settings rows: 18

## Face usage

- face_001: baseline_reference; no prompt restriction detected
- face_002: primary; no prompt restriction detected
- face_003: primary; no prompt restriction detected
- face_004: primary; no prompt restriction detected
- face_005: primary; no prompt restriction detected
- face_006: primary; no prompt restriction detected
- face_007: secondary_prompt_restricted; no glasses prompts, no beard/stubble prompts
- face_008: secondary_prompt_restricted; no glasses prompts, no beard/stubble prompts

## Selected prompt-type counts

`{'beard': 4, 'smile': 2, 'headphones': 6, 'glasses': 6, 'clothing': 6}`

## Main clean rejection reasons

`{'clean_clip_margin_nonpositive:-0.0032': 1, 'clean_ssim_below_threshold:0.3594': 1, 'clean_ssim_below_threshold:0.3774': 1, 'clean_clip_margin_nonpositive:-0.0093': 1, 'clean_ssim_below_threshold:0.3018': 1, 'clean_clip_margin_nonpositive:-0.0014': 1, 'clean_clip_margin_nonpositive:-0.0067': 1, 'clean_clip_margin_nonpositive:-0.0033': 1, 'clean_ssim_below_threshold:0.3721': 1, 'clean_ssim_below_threshold:0.4385': 1, 'clean_ssim_below_threshold:0.4323': 1, 'clean_clip_margin_nonpositive:-0.0037': 1, 'clean_clip_margin_nonpositive:-0.0005': 1, 'clean_ssim_below_threshold:0.4258': 1, 'clean_clip_margin_nonpositive:-0.0075': 1, 'clean_clip_margin_nonpositive:-0.0111': 1, 'clean_ssim_below_threshold:0.4017': 1, 'clean_ssim_below_threshold:0.4164': 1, 'clean_ssim_below_threshold:0.4371': 1}`

A clean case is selected only when the clean edit has positive CLIP margin and avoids coarse global-collapse checks. Visual audit is still required from the contact sheet.
