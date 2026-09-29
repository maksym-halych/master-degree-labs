.DEFAULT_GOAL := help
.PHONY: help run download-models pre-commit

CACHE_DIR   := ./.cache
MODELS_DIR  := $(CACHE_DIR)/models
HF_REPO     := https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main
VAD_URL     := https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx

# fp32 weights: ~2.5 GB on disk, ~3 GB RSS per worker (measured, not the 6 GB
# once assumed). The int8 variant is 670 MB but its encoder is unusable — it
# decodes Ukrainian lecture audio to ~1 character per second against fp32's 14,
# emitting the blank token at nearly every timestep, and it is no faster on CPU.
# encoder-model.onnx.data holds the external weights and must sit beside the graph.
MODEL_FILES := config.json vocab.txt nemo128.onnx encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

# FILE_ID comes from .env; passing it here expands to a positional argument that
# overrides the setting, and expands to nothing when unset.
run: ## Transcribe the recording in .env, or: make run FILE_ID=<id> [ARGS=-v]
	uv run lecture-transcriber $(FILE_ID) $(ARGS)

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
	uv run pre-commit run --all-files
