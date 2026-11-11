"""Download CQ500 (Chilamkurthy et al., Lancet 2018; CC BY-NC-SA 4.0) and convert each scan to brain-window PNG slices.

    python scripts/prepare_cq500.py [--workers 4]

For each of the 491 scans: download the zip, keep the series closest to 5 mm that is not a bone kernel, convert to
Hounsfield units, apply the brain window (level 40, width 80), save 8-bit PNGs in head-to-foot order, delete the zip.
Resumable: scans that already have PNGs are skipped. Labels come from reads.csv (3 radiologists, majority vote).
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from stroke.data import DATA

URL = "https://s3.ap-south-1.amazonaws.com/qure.headct.study"
OUT = DATA / "external" / "cq500"
LEVEL, WIDTH = 40, 80


def series_score(ds):
    bone = "BONE" in f"{ds.get('ConvolutionKernel', '')} {ds.get('SeriesDescription', '')}".upper()
    return (bone, abs(float(ds.get("SliceThickness", 99) or 99) - 5))


def convert(name):
    dst = OUT / "png" / name
    if dst.exists() and any(dst.iterdir()):
        return name, "skip"
    raw = urllib.request.urlopen(f"{URL}/{name}.zip", timeout=600).read()
    series = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for f in z.namelist():
            if f.lower().endswith(".dcm"):
                ds = pydicom.dcmread(io.BytesIO(z.read(f)))
                series.setdefault(ds.SeriesInstanceUID, []).append(ds)
    if not series:
        return name, "no dicom"
    best = min(series.values(), key=lambda s: series_score(s[0]))
    best.sort(key=lambda d: -float(d.ImagePositionPatient[2]))
    tmp = dst.with_name(name + ".partial")  # renamed when complete, so an interrupted scan is redone
    tmp.mkdir(parents=True, exist_ok=True)
    for i, ds in enumerate(best):
        hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        img = np.clip((hu - (LEVEL - WIDTH / 2)) / WIDTH * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(tmp / f"{i:03d}.png")
    (tmp / "thickness_mm.txt").write_text(str(float(best[0].get("SliceThickness", 0) or 0)))
    tmp.rename(dst)
    return name, f"{len(best)} slices, {best[0].get('SliceThickness')} mm"


def _safe(name):
    try:
        return convert(name)
    except Exception as e:  # one bad scan shouldn't stop the batch; rerun to retry
        return name, f"FAILED {e!r}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reads = OUT / "reads.csv"
    if not reads.exists():
        urllib.request.urlretrieve(f"{URL}/reads.csv", reads)
    df = pd.read_csv(reads)
    labels = pd.DataFrame({"scan": df.name,
                           "ich": (df[["R1:ICH", "R2:ICH", "R3:ICH"]].sum(axis=1) >= 2).astype(int)})
    labels.to_csv(OUT / "labels.csv", index=False)
    names = df.name.tolist()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (name, msg) in enumerate(ex.map(_safe, names)):
            print(f"[{i + 1}/{len(names)}] {name}: {msg}", flush=True)
