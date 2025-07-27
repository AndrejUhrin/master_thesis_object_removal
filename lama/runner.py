"""
Wrapper that lets a notebook do
    from lama.runner import run_lama

Additionaly it It fixes Hydra's errors when non-ASCII by using a sanitised stem only
for the temporary working folder.  The final PNG keeps the original
filename.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Union

from PIL import Image


_ASCII_RE = re.compile(r"[^A-Za-z0-9]+")


def _safe_ascii(stem: str) -> str:
    out = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    out = _ASCII_RE.sub("", out)
    return out or "file"


def _copy_as_png(src: Path, dst: Path, *, is_mask: bool = False) -> None:
    """
    Save *src* to *dst* as PNG **without resizing**.
    If *src* is already a PNG we just copy it.
    """
    if src.suffix.lower() == ".png":
        shutil.copyfile(src, dst)
        return

    img = Image.open(src)
    if not is_mask:
        img = img.convert("RGB")
    img.save(dst, "PNG")


def run_lama(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    model_dir: Union[str, Path],
    out_dir: Union[str, Path],
    python_exec: str | None = None,
    device: str | None = None,         
) -> Path:

    image_path = Path(image_path).expanduser().resolve()
    mask_path = Path(mask_path).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    repo_root = model_dir.parent                    
    predict_py = repo_root / "bin" / "predict.py"
    if not predict_py.exists():
        raise FileNotFoundError(f"{predict_py} not found")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    python_exec = python_exec or sys.executable

    stem_orig = image_path.stem          
    stem_safe = _safe_ascii(stem_orig)   

    with tempfile.TemporaryDirectory(prefix=f"lama_{stem_safe}_") as tmp:
        tmp_dir = Path(tmp)
        img_png = tmp_dir / f"{stem_safe}.png"
        msk_png = tmp_dir / f"{stem_safe}_mask.png"

        _copy_as_png(image_path, img_png)
        _copy_as_png(mask_path, msk_png, is_mask=True)

        cmd = [
            python_exec,
            str(predict_py),
            f"model.path={model_dir}",
            f"indir={tmp_dir}",
            f"outdir={tmp_dir}",
            "dataset.img_suffix=.png",
        ]
        if device:
            cmd.append(f"device={device}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + (
            os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
        )

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        result_png = tmp_dir / msk_png.name
        if not result_png.exists():
            raise RuntimeError("LaMa did not create expected output file")

        final_png = out_dir / f"{stem_orig}.png"   
        shutil.move(result_png, final_png)
        return final_png
