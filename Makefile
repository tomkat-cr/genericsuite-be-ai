# .DEFAULT_GOAL := local
# .PHONY: tests
SHELL := /bin/bash

lock:
	poetry lock

update:
	poetry update

build:
	# Build 'dist' directory needed for the Pypi publish
	poetry lock --no-update
	rm -rf dist
	python3 -m build

publish-test: build
	# Pypi Test publish
	python3 -m twine upload --repository testpypi dist/*

publish: build
	# Production Pypi publish
	python3 -m twine upload dist/*
