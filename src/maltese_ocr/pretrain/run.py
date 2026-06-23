"""Entry point for Stage 1 SeqCLR pretraining.

Usage:
    python -m maltese_ocr.pretrain.run --config configs/stage1.yaml
    python -m maltese_ocr.pretrain.run --config configs/stage1.yaml --max-steps 20
    python -m maltese_ocr.pretrain.run --config configs/stage1.yaml \
        --resume models/stage1/checkpoint_step5000.pt
"""

from __future__ import annotations

import argparse
import logging

import yaml

from maltese_ocr.pretrain.seqclr import SeqCLRTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the stage1 YAML config.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (overrides total_steps; truncated runs "
        "skip the linear probe).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a full checkpoint (model + optimizer + scheduler + step).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = SeqCLRTrainer(config)
    if args.resume:
        trainer.resume(args.resume)

    summary = trainer.train(max_steps=args.max_steps)

    print(
        f"Done. final_step={summary['final_step']} "
        f"truncated={summary['truncated']} probe_cer={summary['probe_cer']}"
    )
    if summary["losses"]:
        print(f"first_loss={summary['losses'][0]:.4f} last_loss={summary['losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
