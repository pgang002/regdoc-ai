.PHONY: setup setup-paddle data inventory render baseline populated evaluate-populated degrade evaluate-degradation figures benchmark-ocr benchmark-ocr-strict check-paddle tables table-robustness table-figures test day1 day2 day3 day4 day5 day6 day7 day8 day9 day10 redaction redaction-figures document-data document-classifier document-understanding document-figures final-scorecard monitoring-validation deployment-validation

setup:
	python -m pip install -e .

setup-paddle:
	python -m pip install -r requirements-paddle.txt

data:
	python scripts/download_fda_forms.py

inventory:
	python scripts/build_data_manifest.py
	python scripts/inspect_fda_forms.py

render:
	python scripts/render_fda_forms.py --dpi 150

baseline:
	python scripts/run_tesseract_baseline.py --modes raw clahe --psm 6

populated:
	python scripts/extract_protocol_metadata.py
	python scripts/generate_populated_forms.py

evaluate-populated:
	python scripts/evaluate_populated_forms.py

degrade:
	python scripts/generate_degraded_forms.py

evaluate-degradation:
	python scripts/evaluate_degradation_condition.py clean
	python scripts/evaluate_degradation_condition.py rotation_2deg
	python scripts/evaluate_degradation_condition.py gaussian_blur
	python scripts/evaluate_degradation_condition.py gaussian_noise
	python scripts/evaluate_degradation_condition.py low_contrast
	python scripts/evaluate_degradation_condition.py directional_shadow
	python scripts/evaluate_degradation_condition.py jpeg_compression
	python scripts/evaluate_degradation_condition.py combined_moderate
	python scripts/summarize_degradation_benchmark.py

figures:
	python scripts/create_degradation_figures.py

check-paddle:
	python scripts/check_paddleocr_runtime.py

benchmark-ocr:
	python scripts/benchmark_ocr_engines.py --engines tesseract paddleocr

benchmark-ocr-strict:
	python scripts/benchmark_ocr_engines.py --engines tesseract paddleocr --strict

tables:
	python scripts/build_protocol_table_benchmark.py
	python scripts/evaluate_table_extraction.py

table-robustness:
	python scripts/generate_degraded_tables.py
	python scripts/evaluate_table_robustness.py

table-figures:
	python scripts/create_table_figures.py

test:
	pytest

day1: data inventory render baseline test

day2: populated evaluate-populated test

day3: degrade evaluate-degradation figures test

day4: benchmark-ocr test

day5: tables table-robustness table-figures test


document-data:
	python scripts/build_document_understanding_benchmark.py

document-classifier:
	python scripts/train_document_classifier.py

document-understanding:
	python scripts/evaluate_document_understanding.py

document-figures:
	python scripts/create_document_understanding_figures.py

day6: document-data document-classifier document-understanding document-figures test


redaction:
	python scripts/evaluate_redaction.py

redaction-figures:
	python scripts/create_redaction_figures.py

day7: redaction redaction-figures test


api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
	streamlit run app/streamlit_app.py --server.port 8501

app-integration:
	python scripts/run_day8_integration.py

app-figures:
	python scripts/create_day8_figures.py

day8: app-integration app-figures test


infra-setup:
	python -m pip install -r requirements-infra.txt

db-init:
	python scripts/init_database.py

worker:
	celery -A regdoc_ai.worker.celery_app:celery_app worker --loglevel=INFO --queues=documents --concurrency=2

day9-integration:
	python scripts/run_day9_integration.py
	python scripts/create_day9_figures.py

day9: day9-integration test

compose-up:
	docker compose up --build

compose-down:
	docker compose down


final-scorecard:
	python scripts/create_final_scorecard.py
	python scripts/create_day10_architecture.py

monitoring-validation:
	python scripts/run_day10_validation.py

deployment-validation:
	python scripts/validate_deployment.py

day10: final-scorecard deployment-validation monitoring-validation test
