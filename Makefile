.DEFAULT_GOAL := help
.PHONY: help run download-models pre-commit
.PHONY: render-report run-lab

CACHE_DIR   := ./.cache
MODELS_DIR  := $(CACHE_DIR)/models
# The tool is one project in this repo; .cache/, docs/ and logs/ are shared and
# stay at the root. `--project` points uv at the subdir without changing the
# working directory, so the pipeline's relative paths still land in those trees.
# `--directory` (which does cd) would break them — use it only for dev tools.
PROJECT     := lecture-transcriber
HF_REPO     := https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main
VAD_URL     := https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx

# fp32 weights: ~2.5 GB on disk, ~3 GB RSS per worker (measured, not the 6 GB
# once assumed). The int8 variant is 670 MB but its encoder is unusable — it
# decodes Ukrainian lecture audio to ~1 character per second against fp32's 14,
# emitting the blank token at nearly every timestep, and it is no faster on CPU.
# encoder-model.onnx.data holds the external weights and must sit beside the graph.
MODEL_FILES := config.json vocab.txt nemo128.onnx encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

# FILE_ID comes from .env; passing it here expands to a positional argument that
# overrides the setting, and expands to nothing when unset.
run: ## Transcribe the recording in .env, or: make run FILE_ID=<id> [ARGS=-v]
	uv run --project $(PROJECT) lecture-transcriber $(FILE_ID) $(ARGS)

download-models: $(addprefix $(MODELS_DIR)/,$(MODEL_FILES)) $(MODELS_DIR)/silero_vad.onnx ## Download Parakeet TDT v3 int8 ONNX weights into ./.cache/models
	@echo "models ready in $(MODELS_DIR)"

# Downloads are idempotent: each file is a target, so a re-run only fetches what is missing.
$(MODELS_DIR)/silero_vad.onnx:
	@mkdir -p $(MODELS_DIR)
	curl -fL --progress-bar -o $@ $(VAD_URL)

$(MODELS_DIR)/%:
	@mkdir -p $(MODELS_DIR)
	curl -fL --progress-bar -o $@ $(HF_REPO)/$*

pre-commit: ## Run every pre-commit hook over the tree
	uv run --directory $(PROJECT) pre-commit run --all-files

# --- Lab reports -----------------------------------------------------------
# No subject is named anywhere below. render-report derives everything from
# REPORT, and producing artifacts is each lab's own run.py, so neither a new
# subject nor a new lab needs a change in this file.
#
# Subject paths hold spaces and Cyrillic. GNU Make splits targets, prerequisites
# and $(dir)/$(notdir) on whitespace, so such a path can be none of those: it
# appears only inside recipes, quoted, where the shell handles it.

# Which report to render, as the path to the directory holding its main.tex:
#   REPORT=src/1th term 2026 Autumn/<subject>/lab1
# Read from .env the way FILE_ID is, so `make render-report` takes no arguments.
# `?=` lets a one-off override win: `make render-report REPORT=<other>`.
REPORT ?= $(shell sed -n 's/^[[:space:]]*REPORT=//p' .env 2>/dev/null | tail -1)

# Submission date in the published filename. Defaults to today; pin it with
# `make render-report DATE=2026-10-02` to overwrite an earlier build in place
# rather than leaving a second PDF behind.
DATE ?= $(shell date +%F)

# -lualatex, not -pdf: under pdfLaTeX the listings package cannot read UTF-8 and
# halts on the Cyrillic in the sources (figure labels in the code listings, and
# the report prose itself) — see preamble.tex.
LATEXMK := latexmk -lualatex -shell-escape -interaction=nonstopmode -halt-on-error -file-line-error

# A lab's artifacts are produced by its own run.py, which declares the stages and
# the outputs; run-lab below only locates and launches it. Both operations are
# uniform across every report, so both live here and neither names a subject.
#
# run.py derives every path from __file__, so it does not care where it is run
# from — but its interpreter does: the subject directory holds the pyproject.toml
# that pins it, hence `--project "$$subject"`. Nothing here is relative to the
# working directory, so unlike the lecture-transcriber `run` target above this is
# safe to invoke from anywhere.
run-lab: ## Produce the artifacts for the lab at REPORT (from .env)
	@set -e; \
	src="$(REPORT)"; src="$${src%/}"; \
	[ -n "$$src" ] || { echo "REPORT is unset. Set it in .env, or pass REPORT=<dir holding run.py>"; exit 1; }; \
	[ -f "$$src/run.py" ] || { echo "no run.py in '$$src'"; exit 1; }; \
	subject="$$(dirname "$$src")"; \
	uv run --project "$$subject" python "$$src/run.py"

# The report is the deliverable, so it lands in docs/; only the LaTeX scratch
# files stay next to the source, in build/. Term, subject and report name all
# come from where REPORT sits, so a new subject needs no change here.
#
# Every path component is derived in the shell, never with Make's $(dir)/$(notdir):
# those split on whitespace, and these paths are full of spaces.
render-report: ## Render the report at REPORT (from .env) into docs/Reports/
	@set -e; \
	src="$(REPORT)"; src="$${src%/}"; \
	[ -n "$$src" ] || { echo "REPORT is unset. Set it in .env, or pass REPORT=<dir holding main.tex>"; exit 1; }; \
	[ -f "$$src/main.tex" ] || { echo "no main.tex in '$$src'"; exit 1; }; \
	root="$$PWD"; \
	name="$$(basename "$$src")"; \
	subject="$$(basename "$$(dirname "$$src")")"; \
	term="$$(basename "$$(dirname "$$(dirname "$$src")")")"; \
	stem="$$(printf '%s' "$$name" | sed 's/[0-9]*$$//')"; \
	num="$$(printf '%s' "$$name" | sed 's/^[^0-9]*//')"; \
	title="$$(printf '%s' "$$stem" | sed 's/^./\U&/')$${num:+ $$num}"; \
	out="$$root/docs/Reports/$$term/$$subject"; \
	( cd "$$src" && $(LATEXMK) -outdir=build -jobname="$$name" main.tex ); \
	mkdir -p "$$out"; \
	cp "$$src/build/$$name.pdf" "$$out/$(DATE) - $$title.pdf"; \
	echo "wrote docs/Reports/$$term/$$subject/$(DATE) - $$title.pdf"
