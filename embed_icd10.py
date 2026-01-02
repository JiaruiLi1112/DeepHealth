import argparse
import csv
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class Icd10Label:
    code: str
    disease: str


_LABEL_RE = re.compile(r"^\s*([A-Z][0-9][0-9][A-Z0-9]{0,2})\s*\((.+)\)\s*$")
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}$")


def _read_labels(labels_path: str, *, strict_codes: bool) -> list[Icd10Label]:
    labels: list[Icd10Label] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = _LABEL_RE.match(line)
            if match is not None:
                code, disease = match.group(1), match.group(2)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) == 1:
                    # Some label lists include non-ICD entries (e.g., "Death").
                    # Treat these as both code and disease.
                    code = parts[0].strip()
                    disease = code
                elif len(parts) == 2:
                    code, disease = parts[0].strip(), parts[1].strip()
                else:
                    raise ValueError(
                        f"Unrecognized label format: {line!r}. "
                        "Expected like 'A00 (cholera)', 'CXX Unknown Cancer', or 'Death'."
                    )
                if disease.startswith("(") and disease.endswith(")"):
                    disease = disease[1:-1].strip()

            if strict_codes and not _CODE_RE.match(code):
                raise ValueError(
                    f"Unrecognized ICD10-like code in label: {line!r} (code={code!r}). "
                    "Re-run without --strict-codes to allow non-ICD labels (e.g., 'Death')."
                )
            labels.append(Icd10Label(code=code, disease=disease))
    if not labels:
        raise ValueError(f"No labels found in {labels_path!r}.")
    return labels


def embed_texts(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    all_embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i: i + batch_size]
            toks = tokenizer(
                batch,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            # Use CLS token embedding (same as original script).
            cls_rep = model(**toks)[0][:, 0, :]
            all_embs.append(cls_rep.detach().cpu().to(torch.float32).numpy())

    return np.concatenate(all_embs, axis=0)


def save_umap_plot(
    embeddings: np.ndarray,
    codes: list[str],
    *,
    out_path: str,
    random_state: int = 42,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "UMAP visualization requires matplotlib. Install it with: pip install matplotlib"
        ) from e

    try:
        import umap
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "UMAP visualization requires umap-learn. Install it with: pip install umap-learn"
        ) from e

    reducer = umap.UMAP(n_components=2, metric="cosine",
                        random_state=random_state)
    coords = reducer.fit_transform(embeddings)

    if len(codes) != coords.shape[0]:
        raise ValueError(
            f"codes length ({len(codes)}) does not match embeddings rows ({coords.shape[0]})."
        )

    groups: list[str] = []
    for code in codes:
        cleaned = code.strip()
        if cleaned.lower() == "death":
            groups.append("Death")
        else:
            groups.append(cleaned[:1].upper() if cleaned else "?")

    group_names = sorted({g for g in groups if g != "Death"})
    cmap = plt.get_cmap("tab20")
    group_to_color: dict[str, object] = {
        g: cmap(i % cmap.N) for i, g in enumerate(group_names)
    }
    group_to_color["Death"] = "grey"
    colors = [group_to_color.get(g, "black") for g in groups]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.7, c=colors)
    ax.set_title("UMAP of ICD label embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed ICD-10 disease labels with SapBERT")
    parser.add_argument(
        "--labels",
        default="labels.csv",
        help="Path to labels.csv (lines like 'A00 (cholera)')",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for embeddings and metadata",
    )
    parser.add_argument(
        "--model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="HuggingFace model name",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=25)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--strict-codes",
        action="store_true",
        help="Fail if a label code is not ICD10-like (disallows labels like 'Death')",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Also save a 2D UMAP scatterplot of the embeddings",
    )
    parser.add_argument(
        "--umap-out",
        default=None,
        help="Path to save UMAP PNG (default: <out-dir>/icd10_sapbert_umap.png)",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=42,
        help="Random seed for UMAP",
    )
    args = parser.parse_args()

    labels = _read_labels(args.labels, strict_codes=args.strict_codes)
    texts = [lbl.disease for lbl in labels]
    embs = embed_texts(
        texts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    embs_path = os.path.join(args.out_dir, "icd10_sapbert_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "icd10_sapbert_metadata.tsv")

    np.save(embs_path, embs)

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["index", "icd10_code", "disease"])
        for i, lbl in enumerate(labels):
            w.writerow([i, lbl.code, lbl.disease])

    if args.umap:
        umap_path = (
            args.umap_out
            if args.umap_out is not None
            else os.path.join(args.out_dir, "icd10_sapbert_umap.png")
        )
        save_umap_plot(
            embs,
            [lbl.code for lbl in labels],
            out_path=umap_path,
            random_state=args.umap_random_state,
        )
        print(f"Saved UMAP plot:  {umap_path}")

    print(f"Saved embeddings: {embs_path} (shape={embs.shape})")
    print(f"Saved metadata:   {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
