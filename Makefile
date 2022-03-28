SHELL=/bin/bash
PROJECT_NAME=algos
PROJECT_PATH=${PROJECT_NAME}/
LINT_PATHS=${PROJECT_PATH} experiments/ utilities/

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

pytest:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	pytest tests --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing

mypy:
	$(call check_install, mypy)
	mypy ${PROJECT_PATH} ff_scenarios

lint:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)
	flake8 ${LINT_PATHS} --count --show-source --statistics

format:
	# sort imports
	$(call check_install, isort)
	isort ${LINT_PATHS}
	# reformat using yapf
	$(call check_install, yapf)
	yapf -ir ${LINT_PATHS}

check-codestyle:
	$(call check_install, isort)
	$(call check_install, yapf)
	isort --check ${LINT_PATHS} && yapf -r -d ${LINT_PATHS}

check-docstyle:
	$(call check_install, pydocstyle)
	pydocstyle ${PROJECT_PATH} --convention=google

doc:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	cd docs && make html && cd _build/html && python3 -m http.server

spelling:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)
	cd docs && make spelling SPHINXOPTS="-W"

clean:
	make clean

commit-checks: format lint mypy check-docstyle

# Build docker images
docker:
	./scripts/build_docker.sh ${PROJECT_NAME} interactive

docker-push:
	./scripts/build_docker.sh ${PROJECT_NAME}
	./scripts/tag_and_push.sh ${PROJECT_NAME}

k8s:
	/mnt/home/aiops/tools/make_k8s/build_k8s.sh -p ${PROJECT_NAME}

submit-k8s:
	./scripts/submit_job.sh

run:
	./scripts/create_job.sh $(cmd)

# PyPI package release
release:
	$(call check_install, twine)
	python setup.py sdist bdist_wheel
	twine -r devpi-dev upload dist/*

nightly-upload:
	$(call check_install, twine)
	./scripts/nightly_upload.py ${PROJECT_NAME}

.PHONY: clean spelling doc mypy lint format check-codestyle check-docstyle commit-checks k8s submit-k8s
