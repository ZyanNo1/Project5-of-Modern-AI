import argparse
import json
import subprocess
from pathlib import Path


def load_hparams(run_dir: Path) -> dict:
    hp_path = run_dir / "hparams.json"
    if hp_path.exists():
        return json.loads(hp_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing hparams.json: {hp_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run_dir", type=str, required=True, help="Path to outputs/run_XXXX_XXXXXX")
    ap.add_argument("--python", type=str, default="python")
    ap.add_argument("--train_py", type=str, default="src/train.py")
    ap.add_argument("--evaluate_py", type=str, default="src/evaluate.py")
    ap.add_argument("--do_eval", action="store_true", help="Also run internal-test evaluation for each ablation.")
    args = ap.parse_args()

    base_run = Path(args.base_run_dir).resolve()
    splits = base_run / "splits"
    if not splits.exists():
        raise FileNotFoundError(f"Missing splits dir: {splits}")

    hp = load_hparams(base_run)

    # Use same hyperparams as base run for fairness, but allow overriding by editing hp or CLI.
    common = [
        "--split_dir", str(splits),
        "--data_dir", str(hp.get("data_dir", "./data/data")),
        "--train_txt", str(hp.get("train_txt", "./data/train.txt")),
        "--clip_model", str(hp.get("clip_model", "openai/clip-vit-base-patch32")),
        "--image_size", str(hp.get("image_size", 224)),
        "--max_text_len", str(hp.get("max_text_len", 77)),
        "--batch_size", str(hp.get("batch_size", 16)),
        "--num_workers", str(hp.get("num_workers", 4)),
        "--seed", str(hp.get("seed", 42)),
        "--dropout", str(hp.get("dropout", 0.5)),
        "--hidden_dim", str(hp.get("hidden_dim", 512)),
        "--hidden_dim2", str(hp.get("hidden_dim2", 128)),
        "--weight_decay", str(hp.get("weight_decay", 0.05)),
        "--lr_head", str(hp.get("lr_head", 3e-4)),
        "--lr_clip", str(hp.get("lr_clip", 5e-6)),
        "--lr_attn", str(hp.get("lr_attn", 1e-4)),
        "--fusion", str(hp.get("fusion", "concat")),
        "--early_stop_patience", str(hp.get("early_stop_patience", 3)),
        "--epochs", str(hp.get("epochs", 20)),
        "--stage1_epochs", str(hp.get("stage1_epochs", 5)),
        "--stage2_epochs", str(hp.get("stage2_epochs", 5)),
        "--val_size", str(hp.get("val_size", 0.1)),
        "--test_size", str(hp.get("test_size", 0.1)),
    ]

    flags = []
    if hp.get("two_stage", True):
        flags.append("--two_stage")
    if hp.get("freeze_clip", False):
        flags.append("--freeze_clip")
    if hp.get("unfreeze_text", False):
        flags.append("--unfreeze_text")
        flags.extend(["--unfreeze_text_layers", str(hp.get("unfreeze_text_layers", 1))])
    if hp.get("unfreeze_vision", False):
        flags.append("--unfreeze_vision")
        flags.extend(["--unfreeze_vision_layers", str(hp.get("unfreeze_vision_layers", 1))])
    if hp.get("class_weights", False):
        flags.append("--class_weights")

    modes = ["multimodal", "text_only", "image_only"]

    for mode in modes:
        out_dir = base_run / "ablations" / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [args.python, args.train_py, "--run_dir", str(out_dir), "--input_mode", mode] + common + flags
        print("\n=== Train:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        if args.do_eval:
            ckpt = out_dir / "best_checkpoint.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

            eval_cmd = [
                args.python, args.evaluate_py,
                "--split_dir", str(splits),
                "--data_dir", str(hp.get("data_dir", "./data/data")),
                "--checkpoint", str(ckpt),
                "--batch_size", str(hp.get("batch_size", 16)),
                "--num_workers", str(hp.get("num_workers", 4)),
                "--image_size", str(hp.get("image_size", 224)),
                "--max_text_len", str(hp.get("max_text_len", 77)),
            ]
            print("\n=== Eval:", " ".join(eval_cmd))
            subprocess.run(eval_cmd, check=True)

    print("\nDone. Ablations saved under:", base_run / "ablations")


if __name__ == "__main__":
    main()