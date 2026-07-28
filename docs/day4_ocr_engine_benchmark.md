# Day 4: Tesseract versus PaddleOCR benchmark harness

## Objective

Day 4 adds a controlled OCR-engine comparison without changing the actual-data basis or
field selection established in Days 2 and 3. Tesseract and PaddleOCR are required to
receive the same:

- official FDA Forms 1572, 3454, and 3455;
- public Moderna protocol metadata embedded in the forms;
- controlled test identifiers where real personal data would be inappropriate;
- 20 selected fields per scan condition;
- eight conditions: clean, rotation, blur, noise, low contrast, shadow, JPEG compression,
  and combined degradation;
- document-level deskew and restoration;
- PDF-coordinate field crops; and
- schema-aware post-OCR validation.

The benchmark deliberately excludes checkbox results because checkbox state is produced
by the classical computer-vision detector, not the text recognizer being compared.

## Implementation

The new engine abstraction is in `src/regdoc_ai/ocr/engines.py`.

- `TesseractFieldEngine` wraps the existing Tesseract pipeline and retains PSM 6 so its
  settings remain comparable with the prior milestones.
- `PaddleOCRFieldEngine` lazily initializes PaddleOCR's English PP-OCRv5 mobile text
  recognition model. Lazy loading keeps the core project operational when the optional
  Paddle runtime is absent.
- Multi-line fields are segmented by horizontal ink projections before Paddle text-line
  recognition; output is reassembled in top-to-bottom order.
- Paddle result parsing accepts common PaddleOCR 3.x result-object and dictionary shapes,
  with tests covering nested and plural outputs.

The reproducible entry point is:

```bash
python scripts/benchmark_ocr_engines.py --engines tesseract paddleocr --strict
```

The script never substitutes an available engine for a missing one. Each engine receives
its own result directory, and `results/ocr_engine_benchmark/benchmark_status.json`
records `completed`, `unavailable`, or `failed` separately.

## Current execution status

The complete Tesseract arm was executed in this environment on all 160 field instances.
The PaddlePaddle/PaddleOCR binary packages and model assets could not be downloaded from
this sandbox, so PaddleOCR was recorded as unavailable and no Paddle metrics were
fabricated. The executable Google Colab notebook is included at
`notebooks/04_tesseract_vs_paddleocr.ipynb`.

### Tesseract result on the controlled Day 4 input

| Metric | Result |
|---|---:|
| Field instances | 160 |
| Raw exact-match accuracy | 88.13% |
| Validated exact-match accuracy | 96.88% |
| Mean character error rate | 0.0910 |
| Mean OCR confidence | 90.62 |
| Mean field OCR latency | 0.1234 seconds |
| Total field OCR latency | 19.75 seconds |

The result includes the clean condition. The clean, blur, noise, JPEG, low-contrast, and
rotation conditions reached 100% validated exact match; directional shadow reached 90%,
and combined degradation reached 85%.

These values are measured Tesseract results, not a Tesseract-versus-Paddle conclusion.
No comparative resume claim should be made until PaddleOCR completes on the same data.

## Files added

- `configs/ocr_engine_benchmark.yaml`
- `requirements-paddle.txt`
- `src/regdoc_ai/ocr/engines.py`
- `scripts/benchmark_ocr_engines.py`
- `scripts/check_paddleocr_runtime.py`
- `notebooks/04_tesseract_vs_paddleocr.ipynb`
- `tests/test_ocr_engines.py`
- `results/ocr_engine_benchmark/`

## Reproduce locally or in Colab

```bash
pip install -e .
pip install -r requirements-paddle.txt
python scripts/generate_degraded_forms.py
python scripts/benchmark_ocr_engines.py --engines tesseract paddleocr --strict
pytest
```

If only the Tesseract arm is needed:

```bash
python scripts/benchmark_ocr_engines.py --engines tesseract
```
