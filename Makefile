# .DEFAULT_GOAL := local
.PHONY: lock update requirements lock-rebuild build publish-test publish dev-prepare-local dev-prepare-git dev-prepare-pypi dev-prepare-publish
SHELL := /bin/bash

lock:
	poetry lock

update:
	poetry update

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

lock-rebuild:
	poetry lock --no-update
	poetry install --sync

build:
	# Build 'dist' directory needed for the Pypi publish
	poetry lock --no-update
	rm -rf dist
	python3 -m build

publish-test: dev-prepare-publish requirements build
	# Pypi Test publish
	python3 -m twine upload --repository testpypi dist/*

publish: dev-prepare-publish requirements build
	# Production Pypi publish
	python3 -m twine upload dist/*

dev-prepare-local:
	poetry add --group dev ../genericsuite-be

dev-prepare-git:
	poetry add --group dev git+https://github.com/tomkat-cr/genericsuite-be

dev-prepare-pypi:
	poetry add --group dev genericsuite

dev-prepare-publish:
	if ! poetry remove genericsuite; then echo "'genericsuite' was not removed..."; else "'genericsuite' removed successfully..."; fi;
