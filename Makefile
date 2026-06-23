PYTHON := python3
PIP    := pip3

.PHONY: setup test test-all render-sample pretrain pretrain-smoke train eval package sync-vault

setup:
	$(PIP) install -e ".[dev]"
	pre-commit install

test:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-all:
	$(PYTHON) -m pytest tests/ -v

render-sample:
	$(PYTHON) scripts/render_sample.py

pretrain:
	$(PYTHON) -m maltese_ocr.pretrain.run --config configs/stage1.yaml

# Fast end-to-end smoke of the real SeqCLR loop: tiny batch, 2 steps, CPU-friendly.
# Runs offline once the base model is cached; no HF login needed (corpus falls
# back to data/texts.json). Writes throwaway checkpoints to models/stage1_smoke/.
pretrain-smoke:
	HF_HUB_OFFLINE=1 $(PYTHON) -m maltese_ocr.pretrain.run \
		--config configs/stage1_smoke.yaml --max-steps 2

train:
	$(PYTHON) -m maltese_ocr.train.run --config configs/stage2.yaml
	$(PYTHON) -m maltese_ocr.train.run --config configs/stage3.yaml

eval:
	$(PYTHON) test_baseline.py
	$(PYTHON) scripts/sync_vault.py

# Re-stamp the gitignored Obsidian vault's CER from results.json.
# No-op (exit 0) when the vault is absent, so it is safe on any clone.
sync-vault:
	$(PYTHON) scripts/sync_vault.py

package:
	$(PYTHON) scripts/package_model.py
