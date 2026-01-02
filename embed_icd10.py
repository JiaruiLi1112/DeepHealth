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


def _read_labels(labels_path: str) -> list[Icd10Label]:
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
                if len(parts) != 2:
                    raise ValueError(
                        f"Unrecognized label format: {line!r}. "
                        "Expected like 'A00 (cholera)' or 'CXX Unknown Cancer'."
                    )
                code, disease = parts[0].strip(), parts[1].strip()
                if disease.startswith("(") and disease.endswith(")"):
                    disease = disease[1:-1].strip()

            if not _CODE_RE.match(code):
                raise ValueError(
                    f"Unrecognized ICD10-like code in label: {line!r} (code={code!r})."
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
    args = parser.parse_args()

    labels = _read_labels(args.labels)
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

    print(f"Saved embeddings: {embs_path} (shape={embs.shape})")
    print(f"Saved metadata:   {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
