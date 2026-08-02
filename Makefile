# .DEFAULT_GOAL := local
.PHONY: lock update requirements lock-rebuild build publish-test publish dev-prepare-local dev-prepare-git dev-prepare-pypi dev-prepare-publish
SHELL := /bin/bash

# default show this file
all:
	@cat Makefile

install:
	poetry install

install-dev:
	poetry install --with dev

lock:
	poetry lock

update:
	poetry update

test:
	APP_DB_URI=fake_db_uri APP_DB_ENGINE=MONGODB APP_DB_NAME=mongo APP_NAME=test_app APP_STAGE=test APP_HOST_NAME=localhost APP_SECRET_KEY=fake_secret_key  STORAGE_URL_SEED=xyz APP_SUPERADMIN_EMAIL=fake_email GIT_SUBMODULE_LOCAL_PATH=fake_path CLOUD_PROVIDER=aws AWS_REGION=us-east-1 GET_SECRETS_ENABLED=0 CURRENT_FRAMEWORK=fastapi poetry run pytest .

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

lock-rebuild:
	poetry lock
	# poetry lock --no-update
	poetry install --sync

build:
	# Build 'dist' directory needed for the Pypi publish
	poetry lock
	rm -rf dist
	poetry run python3 -m build

sast-test: requirements
	# bash node_modules/genericsuite-be-scripts/scripts/sast_test.sh
	snyk auth
	snyk code test --severity-threshold=high --all-projects .
	snyk test --severity-threshold=high --all-projects .

publish-test: dev-prepare-publish sast-test build
	# Pypi Test publish
	poetry run python3 -m twine upload --repository testpypi dist/*

publish: dev-prepare-publish requirements build
	# Production Pypi publish
	poetry run python3 -m twine upload dist/*

dev-prepare-local:
	poetry add --group dev ../genericsuite-be

dev-prepare-git:
	poetry add --group dev git+https://github.com/tomkat-cr/genericsuite-be

dev-prepare-pypi:
	poetry add --group dev genericsuite

dev-prepare-publish:
	# if ! poetry remove genericsuite; then echo "'genericsuite' was not removed..."; else "'genericsuite' removed successfully..."; fi;
