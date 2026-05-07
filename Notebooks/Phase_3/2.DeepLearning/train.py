from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nltk
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data_utils import (
    TextClassificationDataset,
    build_label_maps,
    build_vocab,
    collate_batch,
    labels_to_ids,
    load_csv_split,
    load_fasttext_embeddings,
    load_glove_embeddings,
    set_seed,
)
from models import BiLSTMAttention, TextCNN


_PRETRAINED_LOADERS = {
    "glove": load_glove_embeddings,
    "fasttext": load_fasttext_embeddings,
}


DEFAULT_LABEL_ORDER = ["Normal", "Depression", "Suicidal", "Anxiety"]
DEFAULT_CLASS_WEIGHTS = [0.68, 0.85, 1.10, 2.21]


def _project_clean_dir() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = p / "Data" / "Clean"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError(
        "Cannot find Data/Clean walking up from train.py. Clone the full repo or set CSV paths in make_args."
    )


def make_args(model: str, **overrides: object) -> argparse.Namespace:
    clean = _project_clean_dir()
    defaults: dict = dict(
        model=model,
        train_csv=str(clean / "train_clean.csv"),
        val_csv=str(clean / "val_clean.csv"),
        test_csv=str(clean / "test_clean.csv"),
        pretrained_path="",
        pretrained_format="glove",
        embed_dim=300,
        hidden_dim=128,
        max_len=256,
        min_freq=2,
        batch_size=32,
        epochs=40,
        lr=1e-3,
        weight_decay=0.01,
        dropout=0.4,
        num_filters=128,
        early_stop_patience=6,
        early_stop_min_delta=0.0,
        lr_patience=2,
        scheduler_factor=0.5,
        clip_norm=1.0,
        seed=42,
        num_workers=0,
        output_dir="dl_outputs",
        no_class_weights=False,
        use_tqdm=True,
        log_train_metric_each_epoch=True,
        checkpoint_dir="",
        checkpoint_every_n_epochs=0,
        checkpoint_on_improve=True,
        save_plots=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def parse_args() -> argparse.Namespace:
    clean = _project_clean_dir()
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bilstm", "textcnn"], required=True)
    p.add_argument("--train_csv", type=str, default=str(clean / "train_clean.csv"))
    p.add_argument("--val_csv", type=str, default=str(clean / "val_clean.csv"))
    p.add_argument("--test_csv", type=str, default=str(clean / "test_clean.csv"))
    p.add_argument("--glove_path", type=str, default="")
    p.add_argument("--pretrained_path", type=str, default="")
    p.add_argument(
        "--pretrained_format",
        type=str,
        choices=("glove", "fasttext"),
        default="glove",
    )
    p.add_argument("--embed_dim", type=int, default=300)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--num_filters", type=int, default=128)
    p.add_argument("--early_stop_patience", type=int, default=6)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--lr_patience", type=int, default=2)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="dl_outputs")
    p.add_argument("--no_class_weights", action="store_true")
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--no_log_train_metric", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--checkpoint_every_n_epochs", type=int, default=0)
    p.add_argument("--no_checkpoint_on_improve", action="store_true")
    p.add_argument("--no_save_plots", action="store_true")
    ns = p.parse_args()
    ns.use_tqdm = not ns.no_tqdm
    ns.log_train_metric_each_epoch = not ns.no_log_train_metric
    ns.checkpoint_on_improve = not ns.no_checkpoint_on_improve
    ns.save_plots = not ns.no_save_plots
    return ns


def ensure_nltk() -> None:
    for r in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r, quiet=True)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_y: list[int] = []
    all_p: list[int] = []
    for batch in loader:
        x, lens, y = batch
        x = x.to(device)
        lens = lens.to(device)
        y = y.to(device)
        logits = model(x, lens)
        pred = logits.argmax(dim=-1)
        all_y.extend(y.cpu().numpy().tolist())
        all_p.extend(pred.cpu().numpy().tolist())
    macro_f1 = f1_score(all_y, all_p, average="macro")
    return macro_f1, np.array(all_y), np.array(all_p)


def _maybe_tqdm(it, use: bool, total: int | None, desc: str):
    if not use:
        return it
    try:
        from tqdm.auto import tqdm

        return tqdm(it, total=total, desc=desc, leave=False)
    except ImportError:
        return it


def _save_ckpt_state(
    path: Path,
    state_dict: dict,
    word2idx: dict,
    args: argparse.Namespace,
    epoch: int,
    val_f1: float,
    train_f1: float | None,
) -> None:
    payload = {
        "model_state": state_dict,
        "word2idx": word2idx,
        "label_order": DEFAULT_LABEL_ORDER,
        "epoch": epoch,
        "val_macro_f1": val_f1,
        "train_macro_f1": train_f1,
        "args": vars(args),
    }
    torch.save(payload, path)


def _save_training_visualizations(
    out_dir: Path,
    model_name: str,
    history: dict[str, list],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    hist_json: dict[str, list] = {}
    for k, v in history.items():
        row: list = []
        for x in v:
            if isinstance(x, float) and np.isnan(x):
                row.append(None)
            elif isinstance(x, np.floating):
                row.append(float(x))
            elif isinstance(x, np.integer):
                row.append(int(x))
            else:
                row.append(x)
        hist_json[k] = row
    with (out_dir / f"{model_name}_history.json").open("w", encoding="utf-8") as f:
        json.dump(hist_json, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed; skipped plots/history json kept", flush=True)
        return

    epochs = history["epoch"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(epochs, history["train_loss"], color="C0")
    axes[0].set_ylabel("train loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_macro_f1"], label="val macro F1", color="C1")
    tmf = history.get("train_macro_f1", [])
    if tmf and not all(
        x is None or (isinstance(x, float) and np.isnan(x)) for x in tmf
    ):
        plot_tf = [
            np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
            for x in tmf
        ]
        axes[1].plot(epochs, plot_tf, label="train macro F1", color="C0")
    axes[1].set_ylabel("macro F1")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history["lr"], color="C2")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"{model_name} training")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{model_name}_learning_curves.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_n = cm.astype(np.float64) / np.maximum(row_sum.astype(np.float64), 1.0)

    fig2, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(cm_n, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("true label")
    ax.set_xlabel("predicted label")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_n[i, j]:.0%})",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig2.colorbar(im, ax=ax, fraction=0.046, label="row-normalized")
    fig2.tight_layout()
    fig2.savefig(plots_dir / f"{model_name}_confusion_matrix_test.png", dpi=150)
    plt.close(fig2)

    f1s = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    fig3, ax3 = plt.subplots(figsize=(6.5, 3.8))
    ax3.bar(class_names, f1s, color="steelblue")
    ax3.set_ylabel("F1 (test)")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        ax3.text(i, min(v + 0.03, 1.0), f"{v:.3f}", ha="center", fontsize=9)
    fig3.tight_layout()
    fig3.savefig(plots_dir / f"{model_name}_per_class_f1_test.png", dpi=150)
    plt.close(fig3)
    print(f"[plots] saved under {plots_dir}", flush=True)


def _resolve_pretrained_path_and_format(args: argparse.Namespace) -> tuple[str, str]:
    p = (getattr(args, "pretrained_path", None) or "").strip()
    if not p:
        p = (getattr(args, "glove_path", None) or "").strip()
    fmt = (getattr(args, "pretrained_format", None) or "glove").strip().lower()
    if fmt not in ("glove", "fasttext"):
        fmt = "glove"
    return p, fmt


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_nltk()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_texts, train_labels_raw = load_csv_split(args.train_csv)
    val_texts, val_labels_raw = load_csv_split(args.val_csv)
    test_texts, test_labels_raw = load_csv_split(args.test_csv)

    n_train = len(train_texts)
    n_val = len(val_texts)
    n_test = len(test_texts)
    print(
        f"[data] train={n_train} val={n_val} test={n_test} device={device}\n"
        f"[data] csv train_csv={args.train_csv}\n"
        f"[data] csv val_csv={args.val_csv}\n"
        f"[data] csv test_csv={args.test_csv}",
        flush=True,
    )

    allowed = set(DEFAULT_LABEL_ORDER)
    for split_name, raw in (
        ("train", train_labels_raw),
        ("val", val_labels_raw),
        ("test", test_labels_raw),
    ):
        extra = set(raw) - allowed
        if extra:
            raise ValueError(
                f"Unknown labels in {split_name}: {sorted(extra)}; allowed={sorted(allowed)}"
            )

    str2id, _ = build_label_maps(DEFAULT_LABEL_ORDER)
    y_train = labels_to_ids(train_labels_raw, str2id)
    y_val = labels_to_ids(val_labels_raw, str2id)
    y_test = labels_to_ids(test_labels_raw, str2id)

    word2idx, _ = build_vocab(train_texts, min_freq=args.min_freq)
    pad_idx = word2idx["<pad>"]
    vocab_size = len(word2idx)

    pretrained_path_str, pretrained_fmt = _resolve_pretrained_path_and_format(args)
    if pretrained_path_str:
        ep = Path(pretrained_path_str)
        if ep.is_file():
            loader_fn = _PRETRAINED_LOADERS[pretrained_fmt]
            pretrained, estats = loader_fn(ep, word2idx, args.embed_dim)
            print(
                "[embed] OK loaded pretrained | "
                f"format={estats['format']} path={estats['path']} "
                f"embed_dim={estats['embed_dim']} vocab={estats['vocab_size']} "
                f"matched={estats['matched_vocab_rows']} "
                f"coverage={float(estats['coverage']):.2%}",
                flush=True,
            )
        else:
            pretrained = None
            kind = "directory" if ep.is_dir() else "missing file"
            print(
                f"[embed] SKIP ({kind}, random init): {ep}",
                flush=True,
            )
    else:
        pretrained = None
        print(
            "[embed] SKIP pretrained_path/glove_path empty (random init)",
            flush=True,
        )

    num_classes = len(DEFAULT_LABEL_ORDER)

    if args.model == "bilstm":
        model = BiLSTMAttention(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            padding_idx=pad_idx,
            dropout=args.dropout,
            pretrained_embedding=pretrained,
        ).to(device)
    else:
        model = TextCNN(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            padding_idx=pad_idx,
            dropout=args.dropout,
            num_filters=args.num_filters,
            filter_sizes=(3, 4, 5),
            pretrained_embedding=pretrained,
        ).to(device)

    if args.no_class_weights:
        crit = nn.CrossEntropyLoss()
    else:
        w = torch.tensor(DEFAULT_CLASS_WEIGHTS, dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.lr_patience,
    )

    train_ds = TextClassificationDataset(train_texts, y_train, word2idx, args.max_len)
    val_ds = TextClassificationDataset(val_texts, y_val, word2idx, args.max_len)
    test_ds = TextClassificationDataset(test_texts, y_test, word2idx, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    best_f1 = -1.0
    patience_left = args.early_stop_patience
    best_state = None
    history_epoch: list[int] = []
    history_loss: list[float] = []
    history_train_f1: list[float] = []
    history_val_f1: list[float] = []
    history_lr: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        batch_it = _maybe_tqdm(
            train_loader,
            args.use_tqdm,
            len(train_loader),
            f"epoch {epoch + 1}/{args.epochs}",
        )
        for batch in batch_it:
            x, lens, y = batch
            x = x.to(device)
            lens = lens.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, lens)
            loss = crit(logits, y)
            loss.backward()
            if args.model == "bilstm":
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        train_f1_epoch: float | None = None
        if args.log_train_metric_each_epoch:
            train_f1_epoch, _, _ = evaluate(model, train_loader, device)

        val_f1, _, _ = evaluate(model, val_loader, device)
        sched.step(val_f1)

        lr_now = opt.param_groups[0]["lr"]
        improved = val_f1 > best_f1 + args.early_stop_min_delta
        if improved:
            best_f1 = val_f1
            patience_left = args.early_stop_patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1

        train_str = f"{train_f1_epoch:.4f}" if train_f1_epoch is not None else "n/a"
        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} train_macro_f1={train_str} "
            f"val_macro_f1={val_f1:.4f} best_val={best_f1:.4f} "
            f"lr={lr_now:.2e} patience={patience_left}/{args.early_stop_patience}",
            flush=True,
        )

        history_epoch.append(epoch + 1)
        history_loss.append(avg_loss)
        history_val_f1.append(val_f1)
        history_lr.append(lr_now)
        if train_f1_epoch is not None:
            history_train_f1.append(float(train_f1_epoch))
        else:
            history_train_f1.append(float("nan"))

        if args.checkpoint_every_n_epochs > 0 and (epoch + 1) % args.checkpoint_every_n_epochs == 0:
            dest = ckpt_root / f"{args.model}_roll_ep{epoch + 1:04d}_val{val_f1:.4f}.pt"
            sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            _save_ckpt_state(
                dest,
                sd_cpu,
                word2idx,
                args,
                epoch + 1,
                val_f1,
                train_f1_epoch,
            )

        if improved and best_state is not None:
            best_path = ckpt_root / f"{args.model}_best.pt"
            _save_ckpt_state(
                best_path,
                best_state,
                word2idx,
                args,
                epoch + 1,
                val_f1,
                train_f1_epoch,
            )
            if args.checkpoint_on_improve:
                dest = ckpt_root / f"{args.model}_improve_ep{epoch + 1:04d}_val{val_f1:.4f}.pt"
                _save_ckpt_state(
                    dest,
                    best_state,
                    word2idx,
                    args,
                    epoch + 1,
                    val_f1,
                    train_f1_epoch,
                )

        if patience_left <= 0:
            print(f"[early stopping] at epoch {epoch + 1}", flush=True)
            break

    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    train_f1_final, y_true_train, y_pred_train = evaluate(model, train_loader, device)
    _, y_true_val, y_pred_val = evaluate(model, val_loader, device)

    infer_batches_start = time.perf_counter()
    test_f1, y_true_test, y_pred_test = evaluate(model, test_loader, device)
    infer_total = time.perf_counter() - infer_batches_start
    n_test_n = len(test_ds)
    avg_ms_per_sample = (infer_total / max(n_test_n, 1)) * 1000.0

    report_train = classification_report(
        y_true_train,
        y_pred_train,
        target_names=DEFAULT_LABEL_ORDER,
        digits=4,
        zero_division=0,
    )
    report_val = classification_report(
        y_true_val,
        y_pred_val,
        target_names=DEFAULT_LABEL_ORDER,
        digits=4,
        zero_division=0,
    )
    report_test = classification_report(
        y_true_test,
        y_pred_test,
        target_names=DEFAULT_LABEL_ORDER,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "model": args.model,
        "best_val_macro_f1": float(best_f1),
        "train_macro_f1_final": float(train_f1_final),
        "val_macro_f1_final": float(f1_score(y_true_val, y_pred_val, average="macro")),
        "test_macro_f1": float(test_f1),
        "train_seconds": train_seconds,
        "test_inference_seconds_total": infer_total,
        "test_avg_ms_per_sample": avg_ms_per_sample,
        "num_test_samples": n_test_n,
        "epochs_run": epoch + 1,
    }

    print("\n--- classification report: train (best weights) ---\n", flush=True)
    print(report_train, flush=True)
    print("\n--- classification report: val ---\n", flush=True)
    print(report_val, flush=True)
    print("\n--- classification report: test ---\n", flush=True)
    print(report_test, flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "word2idx": word2idx,
            "label_order": DEFAULT_LABEL_ORDER,
            "args": vars(args),
            "metrics": metrics,
        },
        out_dir / f"{args.model}_final.pt",
    )
    with (out_dir / f"{args.model}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / f"{args.model}_report_train.txt").open("w", encoding="utf-8") as f:
        f.write(report_train)
    with (out_dir / f"{args.model}_report_val.txt").open("w", encoding="utf-8") as f:
        f.write(report_val)
    with (out_dir / f"{args.model}_report_test.txt").open("w", encoding="utf-8") as f:
        f.write(report_test)

    if args.save_plots:
        _save_training_visualizations(
            out_dir,
            args.model,
            {
                "epoch": history_epoch,
                "train_loss": history_loss,
                "train_macro_f1": history_train_f1,
                "val_macro_f1": history_val_f1,
                "lr": history_lr,
            },
            y_true_test,
            y_pred_test,
            DEFAULT_LABEL_ORDER,
        )


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
