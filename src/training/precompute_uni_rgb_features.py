#!/usr/bin/env python3
"""Precompute frozen UNI-2h RGB features for Orion cell crops.

This script mirrors the Orion crop loading used by ``contrastive_runner`` but
uses a single deterministic UNI preprocessing path and writes chunked feature
files. The resulting cache is intended for training only the mask branch,
fusion layer, and projection/classifier heads without re-running UNI each
epoch.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contrastive_learn_add import HEFusedContrastiveModel
from data.data import CellCropsDataset
from data.orion_data_processing import load_cell_crops_from_orion
from train import load_config


class UniRgbWithMaskTransform:
    """Apply UNI preprocessing to RGB and preserve mask channels by nearest resize."""

    def __init__(self, uni_transform):
        self.uni_transform = uni_transform

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        if x.shape[-1] < 3:
            raise ValueError(f"Expected at least 3 channels, got shape {x.shape}")

        rgb = x[:, :, :3].astype(np.uint8, copy=False)
        extra = x[:, :, 3:]

        rgb_pil = Image.fromarray(rgb)
        rgb_tensor = self.uni_transform(rgb_pil)
        if not torch.is_tensor(rgb_tensor):
            rgb_tensor = torchvision.transforms.functional.to_tensor(rgb_tensor)
        rgb_tensor = rgb_tensor.float()

        if extra.size == 0:
            return rgb_tensor

        extra_tensor = torch.from_numpy(np.ascontiguousarray(extra)).permute(2, 0, 1).float()
        extra_tensor = F.interpolate(
            extra_tensor.unsqueeze(0),
            size=rgb_tensor.shape[-2:],
            mode="nearest",
        ).squeeze(0)
        return torch.cat([rgb_tensor, extra_tensor], dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the Orion training config JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cached feature chunks. Defaults to config rgb_embedding_cache_dir or save_dir/uni_rgb_cache.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for extraction.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config num_workers for extraction.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Number of cells per saved feature chunk.")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for UNI inference.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dtype used when saving rgb_features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an output directory that already contains cache chunks.",
    )
    return parser.parse_args()


def orion_fold_samples(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Use the same deterministic Orion split as contrastive_runner.create_orion_data_loaders."""
    cell_patches_path = config["root_dir"]
    rng = np.random.default_rng(42)
    folders = np.array(glob.glob("CRC*", root_dir=cell_patches_path))
    folders = folders[rng.permutation(len(folders))]

    train_val_samples = folders[:32]
    splits = np.split(train_val_samples, 4)
    val_fold = config.get("orion_val_fold", 3)
    train_samples = np.concatenate([splits[i] for i in range(4) if i != val_fold])
    val_samples = splits[val_fold]
    return train_samples, val_samples


def load_orion_split_crops(config: Dict[str, Any], samples: Iterable[str], split: str):
    cell_patches_path = config["root_dir"]
    mask_name = "cell_masks"
    img_patch_name = "image_patches"
    labels_name = "meta"

    sample_fraction = config.get("orion_sample_fraction", None)
    split_fraction = config.get(f"orion_{split}_sample_fraction", sample_fraction)
    split_max = config.get(f"orion_{split}_max_per_sample", None)
    sample_seed = config.get("orion_sample_seed", 42)
    seed_offset = 0 if split == "train" else 10_000

    crops = []
    for sample_idx, sample in enumerate(samples):
        sample_path = Path(cell_patches_path) / sample
        label_files = glob.glob(str(sample_path / f"{labels_name}_*.csv"))
        sample_crops = load_cell_crops_from_orion(
            str(sample_path),
            mask_name,
            img_patch_name,
            labels_name,
            label_files,
            sample_fraction=split_fraction,
            max_samples=split_max,
            sample_seed=sample_seed + seed_offset + sample_idx,
        )
        crops.extend(sample_crops)
        print(f"[{split}] {sample}: {len(sample_crops)} crops")
    return crops


def make_loader(config: Dict[str, Any], crops, uni_transform, batch_size: int, num_workers: int) -> DataLoader:
    dataset = CellCropsDataset(
        crops=crops,
        transform=UniRgbWithMaskTransform(uni_transform),
        mask=True,
        contrastive=False,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = config.get("precompute_prefetch_factor", 2)
    return DataLoader(dataset, **loader_kwargs)


def flush_chunk(
    output_dir: Path,
    split: str,
    chunk_idx: int,
    feature_parts: List[torch.Tensor],
    label_parts: List[torch.Tensor],
    cell_ids: List[int],
    image_ids: List[str],
    save_dtype: torch.dtype,
) -> Dict[str, Any]:
    features = torch.cat(feature_parts, dim=0).to(dtype=save_dtype)
    labels = torch.cat(label_parts, dim=0).long()
    out_path = output_dir / f"{split}_chunk_{chunk_idx:05d}.pt"
    payload = {
        "rgb_features": features.contiguous(),
        "labels": labels.contiguous(),
        "cell_ids": torch.tensor(cell_ids, dtype=torch.long),
        "image_ids": image_ids,
    }
    torch.save(payload, out_path)
    print(f"[{split}] wrote {out_path} ({features.shape[0]} cells)")
    return {
        "path": out_path.name,
        "n": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "dtype": str(features.dtype).replace("torch.", ""),
    }


def as_int_list(values) -> List[int]:
    out = []
    for value in values:
        if torch.is_tensor(value):
            value = value.item()
        out.append(int(value))
    return out


def as_str_list(values) -> List[str]:
    return [str(v) for v in values]


def extract_split(
    split: str,
    loader: DataLoader,
    model: HEFusedContrastiveModel,
    output_dir: Path,
    chunk_size: int,
    device: torch.device,
    use_amp: bool,
    save_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    feature_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    cell_ids: List[int] = []
    image_ids: List[str] = []
    rows_in_chunk = 0
    chunk_idx = 0
    manifest_chunks = []

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"]
            if isinstance(images, list):
                raise RuntimeError("Precompute loader should not use TwoCropTransform.")

            rgb = images[:, :3].to(device, non_blocking=True)
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                feats = model.rgb_encoder(rgb)

            feats = feats.detach().cpu()
            labels = batch["label"].detach().cpu().long().view(-1)
            feature_parts.append(feats)
            label_parts.append(labels)
            cell_ids.extend(as_int_list(batch["cell_id"]))
            image_ids.extend(as_str_list(batch["image_id"]))
            rows_in_chunk += feats.shape[0]

            if rows_in_chunk >= chunk_size:
                manifest_chunks.append(
                    flush_chunk(
                        output_dir,
                        split,
                        chunk_idx,
                        feature_parts,
                        label_parts,
                        cell_ids,
                        image_ids,
                        save_dtype,
                    )
                )
                feature_parts, label_parts, cell_ids, image_ids = [], [], [], []
                rows_in_chunk = 0
                chunk_idx += 1

            if (batch_idx + 1) % 25 == 0:
                print(f"[{split}] processed {batch_idx + 1}/{len(loader)} batches")

    if rows_in_chunk:
        manifest_chunks.append(
            flush_chunk(
                output_dir,
                split,
                chunk_idx,
                feature_parts,
                label_parts,
                cell_ids,
                image_ids,
                save_dtype,
            )
        )

    return manifest_chunks


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    if not config.get("orion", False):
        raise ValueError("This precompute script currently expects an Orion config with 'orion': true.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config.get("rgb_embedding_cache_dir", Path(config.get("save_dir", ".")) / "uni_rgb_cache"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_chunks = list(output_dir.glob("*_chunk_*.pt"))
    if existing_chunks and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains cache chunks. Pass --overwrite to add/replace files.")
    if existing_chunks and args.overwrite:
        for chunk_path in existing_chunks:
            chunk_path.unlink()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or int(config["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(config["num_workers"])
    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"Using device: {device}")
    print(f"Writing UNI RGB feature cache to: {output_dir}")

    model = HEFusedContrastiveModel(backbone="uni2h", freeze_backbone=True, device=str(device))
    model.rgb_encoder.to(device)
    model.rgb_encoder.eval()

    train_samples, val_samples = orion_fold_samples(config)
    print(f"Train Orion samples: {list(train_samples)}")
    print(f"Val Orion samples: {list(val_samples)}")

    manifest: Dict[str, Any] = {
        "config": str(args.config),
        "root_dir": config["root_dir"],
        "backbone": "uni2h",
        "feature_dim": 1536,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "chunk_size": args.chunk_size,
        "splits": {},
    }

    split_to_samples = {"train": train_samples, "val": val_samples}
    requested_splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in requested_splits:
        crops = load_orion_split_crops(config, split_to_samples[split], split)
        loader = make_loader(config, crops, model.uni_transform, batch_size, num_workers)
        chunks = extract_split(
            split=split,
            loader=loader,
            model=model,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            device=device,
            use_amp=args.amp,
            save_dtype=save_dtype,
        )
        manifest["splits"][split] = {
            "n": int(sum(chunk["n"] for chunk in chunks)),
            "samples": [str(sample) for sample in split_to_samples[split]],
            "chunks": chunks,
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[DONE] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
