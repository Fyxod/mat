## Prompt discovery

- Time: 2026-06-24T05:56:07.647634+00:00
- Selected 6 clean prompt/settings for Phase 1A.

## Manual prompt-selection override

- Time: 2026-06-24T06:15:00+00:00
- Replaced tied automatic selections with six visually audited, localized clean edits. Explicitly excluded cap, beanie, thin-eyeglass, and stubble failures before baseline regeneration.

## Phase 1C semantic scoring

- Time: 2026-06-25T04:34:16.856921+00:00
- CLIP could not be loaded; semantic scoring will mark rows as metric-only fallback instead of strong.

## Phase 1C semantic scoring

- Time: 2026-06-25T04:36:36.338993+00:00
- CLIP could not be loaded; semantic scoring will mark rows as metric-only fallback instead of strong.

## Phase 1C semantic scoring

- Time: 2026-06-25T04:36:36.353777+00:00
- CLIP could not be loaded; semantic scoring will mark rows as metric-only fallback instead of strong.

## Phase 1C parallel smoke

- Time: 2026-06-25T04:37:00.266068+00:00
- Process-level parallel execution completed with 2 worker processes and 2 jobs.

## Phase 1C semantic scoring

- Time: 2026-06-25T04:37:31.780327+00:00
- CLIP could not be loaded; semantic scoring will mark rows as metric-only fallback instead of strong.

## Phase 1C semantic scoring

- Time: 2026-06-25T04:37:32.269425+00:00
- CLIP could not be loaded; semantic scoring will mark rows as metric-only fallback instead of strong.

## Phase 1C semantic preflight

- Time: 2026-06-25T06:03:10.311646+00:00
- CLIP semantic scoring is unavailable: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434. Diagnostics: phase1/outputs/summaries/clip_semantic_diagnostics.json
