# Manual visual audit of prompt discovery

The automatic discovery metric gave tied quality scores because no runtime identity model was enabled. The sheet was manually reviewed before authorizing attacks.

| Status | Prompt / setting | Reason |
| --- | --- | --- |
| Keep | black sunglasses, IGS 1.5 | Clear local edit; source face remains recognizable. |
| Keep | round glasses, IGS 1.5 | Clear local edit; source face remains recognizable. |
| Keep | small earring, IGS 2.0 | Small accessory with stable face identity. |
| Keep | headphones, IGS 1.5 | Stable identity and localized accessory edit. |
| Keep | small beard, IGS 1.5 | Stable facial geometry and semantically successful facial-hair edit. |
| Keep | slight smile, IGS 2.0 | Mild expression change with stable identity. |
| Reject | black baseball cap, IGS 1.0 | Major identity and skin-tone drift. |
| Reject | red beanie, IGS 2.0 | Global red image collapse. |
| Reject | blue beanie, IGS 1.0 | Major identity drift. |
| Reject | thin eyeglasses, IGS 1.0 | Gender/identity drift. |
| Reject | light stubble, IGS 1.0 | Extra-face artifact. |
