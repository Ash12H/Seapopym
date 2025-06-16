.PHONY: lint format test doc publish_test publish

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = Seapopym

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Lint using Ruff
lint:
	poetry run ruff check ./seapopym

## Format using Ruff (ie. equivalent to Black)
format:
	poetry run ruff format ./seapopym

## Generate the documentation using Sphinx
doc:
## Check if pandoc is installed, needed for markdown to rst conversion in Notebooks
	@command -v pandoc >/dev/null 2>&1 || { echo >&2 "Pandoc is not installed. Please install it to proceed."; exit 1; }
	poetry export -f requirements.txt --with doc --output docs/requirements.txt
	poetry run sphinx-apidoc seapopym -o docs/source
	poetry run sphinx-build -b html docs/source/ docs/build/html

## Generate the test package on TestPyPI : https://stackoverflow.com/a/72524326
publish_test:
	poetry build
	poetry publish -r test-pypi

## Generate the official package on PyPI
publish:
	poetry build
	poetry publish

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')