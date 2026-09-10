#!/usr/bin/env python3
"""Build a compact SPC SFCOA-HRRR HDF5 byte index for GitHub Pages.

Normal scheduled use:
    python build_spc_index.py --auto --output-dir site/spc-index --keep 6

The script:
  * probes NOMADS with an 8-byte Range request to find the newest FINAL file;
  * uses h5py over fsspec random-access HTTP to read HDF5 metadata only;
  * records raw HDF5 chunk byte offsets/sizes and filter metadata;
  * writes one timestamped JSON index into the Pages tree;
  * keeps only the newest N timestamped indexes (default 6);
  * writes latest.json as an alias/copy of the newest retained index.

The 500+ MB NetCDF file is not intentionally downloaded. If NOMADS ever stops
honoring HTTP Range requests, the availability probe fails safely and fsspec/h5py
should not be used against that response.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fsspec
import h5py
import numpy as np
import requests

NOMADS_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/para"
DATASET_DIR = "sfcoa-hrrr"
ANALYSIS = "final"
INDEX_VERSION = 6
DEFAULT_KEEP = 6
DEFAULT_LOOKBACK = 8
INDEX_RE = re.compile(r"^(\d{8})T(\d{2})Z-final\.json$")


def pad2(v: int | str) -> str:
    return str(v).zfill(2)


def source_url(date: str, hour: int) -> str:
    hh = pad2(hour)
    return (
        f"{NOMADS_ROOT}/spc_post.{date}/{DATASET_DIR}/"
        f"spc_post.t{hh}z.sfcoa_hrrr.{ANALYSIS}.nc"
    )


def stamp(date: str, hour: int) -> str:
    return f"{date}T{pad2(hour)}Z"


def index_filename(date: str, hour: int) -> str:
    return f"{stamp(date, hour)}-{ANALYSIS}.json"


def date_hour(dt: datetime) -> tuple[str, int]:
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d"), dt.hour


def jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return jsonable(v.item())
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        try:
            return bytes(v).decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            return bytes(v).hex()
    if isinstance(v, (str, int, float, bool)) or v is None:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(v, (tuple, list)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(val) for k, val in v.items()}
    return str(v)


def attrs_dict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for k, v in obj.attrs.items():
            out[str(k)] = jsonable(v)
    except Exception:
        pass
    return out


def filter_pipeline(ds: h5py.Dataset) -> list[dict[str, Any]]:
    plist = ds.id.get_create_plist()
    names = {
        1: "deflate", 2: "shuffle", 3: "fletcher32",
        4: "szip", 5: "nbit", 6: "scaleoffset",
    }
    out: list[dict[str, Any]] = []
    for i in range(plist.get_nfilters()):
        raw = plist.get_filter(i)
        fid = int(raw[0])
        flags = int(raw[1])
        client_data = [int(x) for x in raw[2]]
        raw_name = raw[3] if len(raw) > 3 else b""
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", "replace")
        out.append({
            "id": fid,
            "name": str(raw_name or names.get(fid, fid)),
            "flags": flags,
            "clientData": client_data,
        })
    return out


def normalize_level(v: Any) -> float:
    x = float(v)
    if abs(x) > 2000:
        x /= 100.0
    return x


def axis_descriptor(values: list[float]) -> dict[str, Any]:
    """Compact a regular coordinate axis while retaining exact endpoints/count."""
    n = len(values)
    if n == 0:
        return {"count": 0, "regular": True, "start": None, "step": None}
    if n == 1:
        return {"count": 1, "regular": True, "start": float(values[0]), "step": 0.0}

    start = float(values[0])
    step = float(values[1] - values[0])
    # HRRR projected x/y are regular. Use a tolerance generous enough for float
    # coordinate representation but tight enough to catch a changed grid.
    scale = max(1.0, abs(step), max(abs(float(v)) for v in values))
    tol = scale * 1e-7
    max_error = 0.0
    for i, v in enumerate(values):
        expected = start + i * step
        max_error = max(max_error, abs(float(v) - expected))
    regular = max_error <= tol
    out = {
        "count": n,
        "regular": regular,
        "start": start,
        "step": step,
        "end": float(values[-1]),
        "maxLinearError": max_error,
    }
    if not regular:
        # Defensive fallback for an unexpected future grid. The Worker will refuse
        # one-request field mode rather than silently reconstruct an incorrect grid.
        out["values"] = values
    return out


def dataset_meta(name: str, ds: h5py.Dataset, isobaric_hpa: list[float]) -> dict[str, Any]:
    shape = [int(x) for x in ds.shape]
    dtype = ds.dtype.str
    itemsize = int(ds.dtype.itemsize)
    meta: dict[str, Any] = {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "elementSize": itemsize,
        "attrs": attrs_dict(ds),
        "filters": filter_pipeline(ds),
    }

    if ds.chunks is None:
        offset = ds.id.get_offset()
        size = int(ds.id.get_storage_size())
        if offset is None or int(offset) < 0 or size <= 0:
            meta.update({"unsupported": True, "error": "Dataset has no addressable contiguous storage"})
            return meta
        meta.update({"storage": "contiguous", "offset": int(offset), "size": size})
        return meta

    chunk_shape = [int(x) for x in ds.chunks]
    meta.update({"storage": "chunked", "chunkShape": chunk_shape})
    chunks: list[dict[str, Any]] = []
    try:
        n = int(ds.id.get_num_chunks())
        for i in range(n):
            info = ds.id.get_chunk_info(i)
            coord = [int(x) for x in info.chunk_offset]
            c: dict[str, Any] = {
                "coord": coord,
                "offset": int(info.byte_offset),
                "size": int(info.size),
                "filterMask": int(info.filter_mask),
            }
            if (
                len(shape) >= 4
                and len(chunk_shape) >= 2
                and len(isobaric_hpa) == shape[1]
                and chunk_shape[1] == 1
                and len(coord) > 1
            ):
                level_index = coord[1]
                if 0 <= level_index < len(isobaric_hpa):
                    c["levelHpa"] = isobaric_hpa[level_index]
            chunks.append(c)
    except Exception as e:
        meta.update({"unsupported": True, "error": f"Could not enumerate raw chunks: {e}"})
        return meta

    chunks.sort(key=lambda c: tuple(c["coord"]))
    meta["chunks"] = chunks
    return meta


def range_probe(url: str, timeout: float = 20.0) -> bool:
    """Confirm an HDF5 file exists and that NOMADS honors byte ranges."""
    try:
        with requests.get(
            url,
            headers={"Range": "bytes=0-7", "User-Agent": "spc-index-pages/1.0"},
            stream=True,
            timeout=timeout,
        ) as r:
            if r.status_code != 206:
                return False
            first = r.raw.read(8)
            return first == b"\x89HDF\r\n\x1a\n"
    except requests.RequestException:
        return False


def find_latest_available(lookback: int) -> tuple[str, int, str] | None:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for back in range(max(0, lookback) + 1):
        dt = now - timedelta(hours=back)
        date, hour = date_hour(dt)
        url = source_url(date, hour)
        print(f"Probe {date} {hour:02d}Z ...", flush=True)
        if range_probe(url):
            return date, hour, url
    return None


def open_hdf5_remote(url: str):
    # 1 MiB readahead reduces metadata round trips. h5py performs seeks; fsspec
    # translates those into HTTP Range reads instead of downloading the full file.
    of = fsspec.open(url, mode="rb", block_size=1024 * 1024, cache_type="readahead")
    fh = of.open()
    return fh, h5py.File(fh, mode="r")


def build_index(date: str, hour: int, local_file: str | None = None) -> dict[str, Any]:
    url = source_url(date, hour)
    fh = None
    if local_file:
        h5 = h5py.File(local_file, "r")
        source = url  # preserve production URL in test output unless explicitly editing below
    else:
        # Re-probe immediately before h5py so a 200/full-file response is not accepted
        # after an upstream behavior change.
        if not range_probe(url):
            raise RuntimeError(f"NOMADS FINAL file is missing or Range is not honored: {url}")
        fh, h5 = open_hdf5_remote(url)
        source = url

    try:
        iso: list[float] = []
        if "isobaric" in h5 and isinstance(h5["isobaric"], h5py.Dataset):
            try:
                iso = [normalize_level(v) for v in np.asarray(h5["isobaric"][...]).reshape(-1)]
            except Exception:
                iso = []

        x: list[float] = []
        y: list[float] = []
        if "x" in h5 and isinstance(h5["x"], h5py.Dataset):
            x = [float(v) for v in np.asarray(h5["x"][...]).reshape(-1)]
        if "y" in h5 and isinstance(h5["y"], h5py.Dataset):
            y = [float(v) for v in np.asarray(h5["y"][...]).reshape(-1)]

        projection = attrs_dict(h5["Lambert_Conformal"]) if "Lambert_Conformal" in h5 else {}

        variables: dict[str, Any] = {}
        for name, obj in h5.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            try:
                variables[str(name)] = dataset_meta(str(name), obj, iso)
            except Exception as e:
                variables[str(name)] = {"name": str(name), "unsupported": True, "error": str(e)}

        names = sorted(variables)
        return {
            "version": INDEX_VERSION,
            "date": date,
            "hour": int(hour),
            "stamp": stamp(date, hour),
            "analysis": ANALYSIS,
            "filename": f"spc_post.t{pad2(hour)}z.sfcoa_hrrr.{ANALYSIS}.nc",
            "sourceUrl": source,
            "indexedAt": datetime.now(timezone.utc).isoformat(),
            "grid": {
                "x": axis_descriptor(x),
                "y": axis_descriptor(y),
                "projection": projection,
            },
            # Retain full x/y for debugging/backward compatibility. Normal one-request
            # browser loads reconstruct them from grid.x/grid.y response headers.
            "x": x,
            "y": y,
            "projection": projection,
            "isobaricHpa": iso,
            "names": names,
            "variables": variables,
        }
    finally:
        h5.close()
        if fh is not None:
            fh.close()


def parse_index_path(path: Path) -> tuple[str, int] | None:
    m = INDEX_RE.match(path.name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def retained_indexes(output_dir: Path) -> list[Path]:
    paths = []
    for p in output_dir.glob("????????T??Z-final.json"):
        if parse_index_path(p):
            paths.append(p)
    return sorted(paths, key=lambda p: p.name)


def current_latest_stamp(output_dir: Path) -> str | None:
    p = output_dir / "latest.json"
    if not p.exists():
        return None
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
        return str(value.get("stamp")) if value.get("stamp") else None
    except Exception:
        return None


def write_pages_index(index: dict[str, Any], output_dir: Path, keep: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date = str(index["date"])
    hour = int(index["hour"])
    target = output_dir / index_filename(date, hour)
    compact = json.dumps(index, separators=(",", ":"), allow_nan=False) + "\n"
    target.write_text(compact, encoding="utf-8")

    all_indexes = retained_indexes(output_dir)
    if keep < 1:
        keep = 1
    for old in all_indexes[:-keep]:
        old.unlink(missing_ok=True)
    kept = retained_indexes(output_dir)
    if not kept:
        raise RuntimeError("No timestamped SPC indexes remained after pruning")

    newest = kept[-1]
    newest_data = json.loads(newest.read_text(encoding="utf-8"))
    (output_dir / "latest.json").write_text(
        json.dumps(newest_data, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Tiny status file is convenient for humans/health checks and does not count as
    # an hourly index. Normal Worker operation uses latest.json directly.
    status = {
        "version": 1,
        "latest": newest_data.get("stamp"),
        "latestFile": newest.name,
        "retained": [p.name for p in kept],
        "count": len(kept),
        "retentionTarget": keep,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    print(f"Wrote {target} ({target.stat().st_size:,} bytes)")
    print(f"latest.json -> {newest.name}")
    print("Retained timestamped indexes:")
    for p in kept:
        print(f"  {p.name}")
    return status


def validate_date_hour(date: str, hour: int) -> None:
    if len(date) != 8 or not date.isdigit() or not 0 <= hour <= 23:
        raise SystemExit("Invalid date/hour; use YYYYMMDD and UTC hour 0-23")


def main() -> None:
    ap = argparse.ArgumentParser()
    source_group = ap.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--auto", action="store_true", help="Find newest available FINAL within --lookback hours")
    source_group.add_argument("--date", help="Explicit YYYYMMDD (requires --hour)")
    ap.add_argument("--hour", type=int, help="Explicit UTC hour 0-23")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Hours back to probe with --auto (default 8)")
    ap.add_argument("--output-dir", default="site/spc-index", help="GitHub Pages output directory")
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="Number of timestamped hourly indexes to retain")
    ap.add_argument("--force", action="store_true", help="Rebuild even if newest candidate is already latest")
    ap.add_argument("--local-file", help="Local HDF5 file for parser testing; requires --date and --hour")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    if args.local_file:
        if not args.date or args.hour is None:
            raise SystemExit("--local-file requires --date and --hour")
        date, hour = args.date, args.hour
        validate_date_hour(date, hour)
    elif args.date:
        if args.hour is None:
            raise SystemExit("--date requires --hour")
        date, hour = args.date, args.hour
        validate_date_hour(date, hour)
    else:
        # Default to --auto for scheduled GitHub Action simplicity.
        found = find_latest_available(args.lookback)
        if not found:
            print(f"No range-readable SPC FINAL found in the last {args.lookback + 1} hours", file=sys.stderr)
            raise SystemExit(0)
        date, hour, _ = found

    wanted_stamp = stamp(date, hour)
    latest_stamp = current_latest_stamp(output_dir)
    timestamped = output_dir / index_filename(date, hour)
    if not args.force and latest_stamp == wanted_stamp and timestamped.exists():
        print(f"NO_CHANGE: {wanted_stamp} is already the published latest index")
        # Still enforce retention if a human has left extra old files in the tree.
        kept = retained_indexes(output_dir)
        for old in kept[:-max(1, args.keep)]:
            old.unlink(missing_ok=True)
        raise SystemExit(0)

    print(f"Building SPC index for {date} {hour:02d}Z")
    idx = build_index(date, hour, args.local_file)
    write_pages_index(idx, output_dir, args.keep)


if __name__ == "__main__":
    main()
