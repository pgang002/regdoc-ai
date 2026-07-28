# Generated degradation images

`images/` contains 64 deterministic derived page images when the benchmark is generated locally.
The compact repository archive may omit that directory because it is approximately 344 MB and can be reproduced exactly from the retained populated-form page images.

Regenerate it with:

```powershell
python scripts/generate_degraded_forms.py
```

`manifest.csv` and `metadata.json` record the source image, output hash, condition, seed, and exact augmentation parameters for every generated page.
