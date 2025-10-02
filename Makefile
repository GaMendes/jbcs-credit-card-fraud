SHELL := /bin/bash

install-deps:
	python -m venv .venv && \
	source .venv/bing/activate && \
	pip install -r requirements.txt

run-experiments:
	python main.py

run-analysis:
	python post_experiment_analysis.py