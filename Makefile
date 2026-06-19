PYTHON := python3
PIP    := pip3

.PHONY: setup test render-sample pretrain train eval package sync-vault

setup:
	$(PIP) install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v -m "not slow"

test-all:
	pytest tests/ -v

render-sample:
	$(PYTHON) scripts/render_sample.py

pretrain:
	$(PYTHON) -m maltese_ocr.pretrain.run --config configs/stage1.yaml

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
