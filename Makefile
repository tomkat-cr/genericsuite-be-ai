# .DEFAULT_GOAL := local
# .PHONY: tests
SHELL := /bin/bash

lock:
	poetry lock

update:
	poetry update

lock-rebuild:
	poetry lock --no-update
	poetry install --sync

build:
	# Build 'dist' directory needed for the Pypi publish
	poetry lock --no-update
	rm -rf dist
	python3 -m build

publish-test: dev-prepare-publish build
	# Pypi Test publish
	python3 -m twine upload --repository testpypi dist/*

publish: dev-prepare-publish build
	# Production Pypi publish
	python3 -m twine upload dist/*

dev-prepare-local:
	poetry add --group dev ../genericsuite-be

dev-prepare-git:
	poetry add --group dev git+https://github.com/tomkat-cr/genericsuite-be

dev-prepare-pypi:
	poetry add --group dev genericsuite

dev-prepare-publish:
	poetry remove genericsuite
