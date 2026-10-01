"""SEN2NEON: real Sentinel-2 10 m / NEON 2.5 m pairs.

This is the only module in the repository that produces *native* training
pairs -- a real 10 m observation next to a real 2.5 m reference of the same
ground. Everything else (`synthetic.py`, the Wald degradation in
`degradation.py`) simulates the LR side, which is useful for smoke tests and
unavoidable when no reference exists, but cannot measure whether a model
recovers detail that a sensor actually saw.

Dataset facts, verified against the Hub on 2026-09-30 rather than assumed:

* ``s2_l2a_10m/``            observed Sentinel-2 L2A, 12 x 256 x 256, nodata 65535
* ``neon_2.5m_linearized/``  NEON AVIRIS-NG reference, 12 x 1024 x 1024, nodata 0
* both uint16 scaled by 10000, pixel-aligned, exactly 4x apart, CC-BY-4.0
* band order is ``B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12`` -- **B10 is absent**,
  so RGB+NIR are bands 4, 3, 2 and 8, *not* the first four. Reading
  ``src.read()[:4]`` silently yields B1/B2/B3/B4 (coastal-aerosol, blue, green,
  red) and trains a model on the wrong spectra. `band_indices` exists to make
  that mistake impossible.

Three properties of the pairs limit what any accuracy number here can mean,
and they are recorded per tile so a report can state them:

* **Temporal separation.** Median 23 days between the NEON overflight and the
  Sentinel-2 acquisition, up to 40. Vegetation and water move in that window,
  so some residual is real change, not model error.
* **Cross-sensor radiometry.** The HR side is AVIRIS-NG convolved with
  Sentinel-2 spectral response functions, not a Sentinel-2 observation.
* **Cloud information is per tile, not per pixel.** The index carries
  ``cloud_score_plus_cdf`` (a scene-level Cloud Score+ CDF, higher = clearer);
  the 12-band stack has no SCL layer, so there is no per-pixel cloud mask to
  apply. Tiles are filtered on the scalar and the limitation is reported --
  see `IndexFilter`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import S2_REFLECTANCE_SCALE

REPO_ID = "isp-uv-es/SEN2NEON"
LR_DIR = "s2_l2a_10m"
HR_DIR = "neon_2.5m_linearized"
METADATA = "metadata.csv"

#: Band order inside every SEN2NEON GeoTIFF, as documented on the dataset card.
#: Note the absence of B10 -- this is why positional indexing is unsafe.
SEN2NEON_BAND_ORDER: tuple[str, ...] = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
)

#: The four bands this project models, in the canonical order used everywhere
#: else in the repo: blue, green, red, near-infrared.
RGB_NIR_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08")

#: Nodata sentinels declared by the dataset. They differ between products.
LR_NODATA = 65535
HR_NODATA = 0

#: Filename stems look like ``2018_MLBS_3__0_2`` = year, NEON site, mosaic
#: number, then the row/column of this crop within that mosaic.
_STEM = re.compile(r"^(?P<year>\d{4})_(?P<site>[A-Z]+)_(?P<mosaic>\d+)__(?P<row>\d+)_(?P<col>\d+)$")


def normalise_band(name: str) -> str:
    """``B4`` -> ``B04``, ``b8a`` -> ``B8A``; leaves already-canonical names alone.

    Sentinel-2 band names are written both zero-padded (``B04``, this project's
    configs) and bare (``B4``, the SEN2NEON card). Comparing the two forms as
    raw strings is a silent mismatch, so every comparison goes through here.

    The one exception is the narrow NIR band, which is officially ``B8A`` and
    never ``B08A``. Padding it would invent a name no product uses and break
    the very lookup this function exists to make reliable.
    """
    text = str(name).strip().upper()
    if not text.startswith("B"):
        raise ValueError(f"not a Sentinel-2 band name: {name!r}")
    body = text[1:]
    if body.isdigit():
        return f"B{int(body):02d}"
    return text


def band_indices(
    wanted: Sequence[str] = RGB_NIR_BANDS,
    available: Sequence[str] = SEN2NEON_BAND_ORDER,
) -> tuple[int, ...]:
    """Map band *names* to 1-based rasterio band indices.

    >>> band_indices(("B04", "B03", "B02", "B08"))
    (4, 3, 2, 8)

    Raises rather than guessing when a band is absent, because the failure it
    prevents (training on B1/B2/B3/B4 believing they are R/G/B/NIR) produces a
    model that looks fine and is spectrally wrong.
    """
    lookup = {normalise_band(name): i for i, name in enumerate(available, start=1)}
    missing = [b for b in wanted if normalise_band(b) not in lookup]
    if missing:
        raise KeyError(
            f"bands {missing} are not in this product. Available: {sorted(lookup)}"
        )
    return tuple(lookup[normalise_band(b)] for b in wanted)


@dataclass(frozen=True)
class TilePair:
    """One aligned LR/HR tile, with the provenance needed to split and report."""

    tile_id: str
    lr_path: str            # relative to the dataset root
    hr_path: str
    site: str               # NEON site code, e.g. "MLBS" -- the split unit
    mosaic: str             # "2018_MLBS_3": crops of one mosaic overlap in context
    year: int
    lon: float
    lat: float
    crs: str
    land_cover: str
    cloud_score: float      # Cloud Score+ CDF, higher = clearer
    hr_nodata_pct: float
    temporal_days: int

    def resolve(self, root: str | Path) -> tuple[Path, Path]:
        root = Path(root)
        return root / self.lr_path, root / self.hr_path

    def exists(self, root: str | Path) -> bool:
        return all(p.is_file() for p in self.resolve(root))


@dataclass
class IndexFilter:
    """Tile admission rules, all recorded in the manifest so a run is reproducible.

    The cloud threshold operates on a *scene-level* score, not a per-pixel
    mask: SEN2NEON's 12-band stack carries no SCL layer. A tile that passes may
    still contain cloud over part of its footprint. This is a stated limitation
    of the dataset, not something this code can mask away.
    """

    min_cloud_score: float = 0.90      # Cloud Score+ CDF; dataset range is 0.86-0.95
    max_hr_nodata_pct: float = 20.0    # NEON flight lines leave large empty wedges
    max_temporal_days: int = 40        # dataset maximum; raise nothing above it

    def accepts(self, pair: TilePair) -> bool:
        return (
            pair.cloud_score >= self.min_cloud_score
            and pair.hr_nodata_pct <= self.max_hr_nodata_pct
            and pair.temporal_days <= self.max_temporal_days
        )


def _parse_stem(tile_id: str) -> dict[str, str]:
    match = _STEM.match(tile_id)
    if not match:
        # Unrecognised naming still has to land somewhere deterministic; the
        # whole id becomes its own group so it cannot straddle two splits.
        return {"year": "0", "site": tile_id, "mosaic": tile_id, "row": "0", "col": "0"}
    return match.groupdict()


def load_index(
    root: str | Path,
    filters: IndexFilter | None = None,
    require_files: bool = False,
) -> list[TilePair]:
    """Read ``metadata.csv`` into `TilePair` records.

    `require_files` restricts the result to pairs actually present on disk,
    which is what the patch builder wants after a partial download.
    """
    import pandas as pd

    root = Path(root)
    path = root / METADATA
    if not path.is_file():
        raise FileNotFoundError(
            f"no {METADATA} at {path}.\n"
            f"Fetch it first:  python scripts/download_sen2neon.py --root {root}"
        )
    frame = pd.read_csv(path)
    filters = filters or IndexFilter()

    pairs: list[TilePair] = []
    for row in frame.itertuples(index=False):
        parts = _parse_stem(str(row.id))
        pair = TilePair(
            tile_id=str(row.id),
            lr_path=str(row.lr),
            # Always the canonical 2.5 m product: `hr_2_5m_path` is explicit,
            # where the legacy `hr` alias is only conventionally the same.
            hr_path=str(getattr(row, "hr_2_5m_path", row.hr)),
            site=parts["site"],
            mosaic=f"{parts['year']}_{parts['site']}_{parts['mosaic']}",
            year=int(parts["year"]),
            lon=float(row.centroid_lon if hasattr(row, "centroid_lon") else row.lon),
            lat=float(row.centroid_lat if hasattr(row, "centroid_lat") else row.lat),
            crs=str(row.crs),
            land_cover=str(getattr(row, "LC_superclass_text", "unknown")),
            cloud_score=float(row.cloud_score_plus_cdf),
            hr_nodata_pct=float(row.neon_nodata_percent),
            temporal_days=int(row.temporal_difference_days),
        )
        if not filters.accepts(pair):
            continue
        if require_files and not pair.exists(root):
            continue
        pairs.append(pair)
    return pairs


def stratified_subset(
    pairs: Sequence[TilePair],
    limit: int,
    seed: int = 1337,
) -> list[TilePair]:
    """Pick `limit` tiles spread as evenly as possible over NEON sites.

    Taking the first N rows of the index would collect a handful of sites in
    alphabetical order, and a geographic split of that is meaningless. This
    round-robins over sites (best tiles first within each) so a small download
    still spans many locations and land covers.
    """
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[TilePair]] = {}
    for pair in pairs:
        by_site.setdefault(pair.site, []).append(pair)
    for site, items in by_site.items():
        # Prefer clear tiles with the most valid reference area and the
        # smallest temporal gap -- the ones whose residuals are most
        # attributable to the model rather than to the dataset.
        items.sort(key=lambda p: (p.hr_nodata_pct, p.temporal_days, -p.cloud_score))

    sites = sorted(by_site)
    rng.shuffle(sites)
    chosen: list[TilePair] = []
    depth = 0
    while len(chosen) < limit:
        added = False
        for site in sites:
            if depth < len(by_site[site]):
                chosen.append(by_site[site][depth])
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break  # exhausted every site
        depth += 1
    return chosen


def geographic_split(
    pairs: Sequence[TilePair],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 1337,
    group_by: str = "site",
) -> dict[str, list[TilePair]]:
    """Split whole geographic groups into train/val/test.

    Tiles named ``..._3__0_2`` and ``..._3__1_2`` are adjacent crops of one
    NEON mosaic: they share a border, the same flight line, the same
    atmosphere and often the same fields. Splitting per tile puts a val patch
    next to its train neighbour and inflates every metric. Splitting per
    *site* is stronger still -- it also separates repeated observations of one
    location across years -- and is what this uses by default.

    Groups are assigned largest-first to whichever split is furthest below its
    target share, which lands close to 70/15/15 without ever cutting a group.
    """
    if not pairs:
        return {"train": [], "val": [], "test": []}
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    key = (lambda p: p.site) if group_by == "site" else (lambda p: p.mosaic)
    groups: dict[str, list[TilePair]] = {}
    for pair in pairs:
        groups.setdefault(key(pair), []).append(pair)

    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)                      # break ties reproducibly, not alphabetically
    names.sort(key=lambda g: len(groups[g]), reverse=True)

    total = len(pairs)
    splits = ("train", "val", "test")
    targets = dict(zip(splits, (f * total for f in fractions)))
    out: dict[str, list[TilePair]] = {s: [] for s in splits}

    for name in names:
        # "Furthest below target" in absolute tiles, so a single huge site
        # cannot starve val and test of every group.
        deficit = {s: targets[s] - len(out[s]) for s in splits}
        pick = max(splits, key=lambda s: deficit[s])
        out[pick].extend(groups[name])

    if not out["val"] or not out["test"]:
        raise ValueError(
            f"only {len(groups)} geographic groups for a {fractions} split; "
            "val or test came out empty. Download more sites, or pass "
            "group_by='mosaic' to split more finely (weaker separation)."
        )
    return out


def split_report(splits: dict[str, list[TilePair]]) -> dict:
    """Summarise a split, including the leakage check that matters."""
    sites = {s: {p.site for p in items} for s, items in splits.items()}
    total = sum(len(v) for v in splits.values()) or 1
    overlaps = {
        f"{a}&{b}": sorted(sites[a] & sites[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "counts": {s: len(v) for s, v in splits.items()},
        "fractions": {s: round(len(v) / total, 4) for s, v in splits.items()},
        "sites": {s: sorted(v) for s, v in sites.items()},
        "n_sites": {s: len(v) for s, v in sites.items()},
        "shared_sites": overlaps,
        "leakage_free": all(not v for v in overlaps.values()),
        "land_cover": {
            s: {lc: sum(1 for p in items if p.land_cover == lc)
                for lc in sorted({p.land_cover for p in items})}
            for s, items in splits.items()
        },
        "temporal_days_median": {
            s: (float(np.median([p.temporal_days for p in items])) if items else None)
            for s, items in splits.items()
        },
    }


def read_tile(
    path: str | Path,
    bands: Sequence[str] = RGB_NIR_BANDS,
    nodata: int | float | None = None,
    available: Sequence[str] = SEN2NEON_BAND_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the named bands as reflectance, plus a per-pixel valid mask.

    Returns ``(array (C, H, W) float32 in 0-1, valid (H, W) bool)``. Invalid
    pixels are the nodata sentinel or any non-finite value, in *any* band --
    a pixel missing its NIR is not usable as a four-band training target.
    """
    import rasterio

    idx = band_indices(bands, available)
    with rasterio.open(str(path)) as src:
        if src.count < max(idx):
            raise ValueError(
                f"{path} has {src.count} bands but band index {max(idx)} was requested; "
                f"this file is not the documented {len(available)}-band product"
            )
        raw = src.read(idx).astype(np.float32)
        declared = src.nodata
    sentinel = nodata if nodata is not None else declared
    valid = np.isfinite(raw).all(axis=0)
    if sentinel is not None:
        valid &= ~(raw == float(sentinel)).any(axis=0)
    array = np.where(valid[None], raw, np.nan) / S2_REFLECTANCE_SCALE
    return array.astype(np.float32), valid


def read_pair(
    pair: TilePair,
    root: str | Path,
    bands: Sequence[str] = RGB_NIR_BANDS,
    scale: int = 4,
) -> dict[str, np.ndarray]:
    """Read one aligned pair and verify the geometry rather than trusting it."""
    import rasterio

    lr_path, hr_path = pair.resolve(root)
    lr, lr_valid = read_tile(lr_path, bands, nodata=LR_NODATA)
    hr, hr_valid = read_tile(hr_path, bands, nodata=HR_NODATA)

    if hr.shape[-1] != lr.shape[-1] * scale or hr.shape[-2] != lr.shape[-2] * scale:
        raise ValueError(
            f"{pair.tile_id}: HR {hr.shape[-2:]} is not {scale}x LR {lr.shape[-2:]}"
        )
    with rasterio.open(str(lr_path)) as a, rasterio.open(str(hr_path)) as b:
        if a.crs != b.crs:
            raise ValueError(f"{pair.tile_id}: CRS mismatch {a.crs} vs {b.crs}")
        # Same upper-left corner to well within one HR pixel, and an exact
        # integer resolution ratio. Anything else means the pair is not aligned
        # and every metric computed from it would be measuring misregistration.
        if max(abs(a.transform.c - b.transform.c), abs(a.transform.f - b.transform.f)) > 0.5:
            raise ValueError(
                f"{pair.tile_id}: origins differ "
                f"({a.transform.c}, {a.transform.f}) vs ({b.transform.c}, {b.transform.f})"
            )
        if abs(a.transform.a / b.transform.a - scale) > 1e-6:
            raise ValueError(
                f"{pair.tile_id}: pixel size ratio {a.transform.a / b.transform.a} != {scale}"
            )
    return {"lr": lr, "hr": hr, "lr_valid": lr_valid, "hr_valid": hr_valid}


# --------------------------------------------------------------------------- #
# patch extraction
# --------------------------------------------------------------------------- #

@dataclass
class PairPatchSpec:
    """Geometry of the extracted pairs.

    Defaults give the 64 -> 256 pairing named in the system specification:
    a 64x64x4 input tensor and a 256x256x4 reference.
    """

    lr_patch: int = 64
    scale: int = 4
    stride: int | None = None          # in LR pixels; defaults to lr_patch
    min_valid_fraction: float = 0.90   # of the HR window
    min_std: float = 0.004             # reflectance; rejects dead-flat crops
    max_flat_fraction: float = 0.20    # but keep a quota, flat ground is real
    shard_size: int = 128

    @property
    def hr_patch(self) -> int:
        return self.lr_patch * self.scale

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.lr_patch


def iter_pair_patches(
    data: dict[str, np.ndarray],
    spec: PairPatchSpec,
    rng: np.random.Generator,
) -> Iterable[dict[str, np.ndarray]]:
    """Yield aligned LR/HR crops whose HR window is mostly valid.

    The LR and HR windows are cut at the same ground location by construction:
    an LR offset of ``y`` corresponds to HR ``y * scale`` exactly, because the
    products are pixel-aligned with an integer ratio (checked in `read_pair`).
    """
    lr, hr = data["lr"], data["hr"]
    hr_valid, lr_valid = data["hr_valid"], data["lr_valid"]
    size, stride, s = spec.lr_patch, int(spec.stride or spec.lr_patch), spec.scale
    _, height, width = lr.shape

    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            hy, hx, hs = y * s, x * s, size * s
            hr_win_valid = hr_valid[hy : hy + hs, hx : hx + hs]
            if hr_win_valid.mean() < spec.min_valid_fraction:
                continue
            if not lr_valid[y : y + size, x : x + size].all():
                continue
            hr_win = hr[:, hy : hy + hs, hx : hx + hs]
            lr_win = lr[:, y : y + size, x : x + size]
            if not np.isfinite(lr_win).all():
                continue
            # Texture is judged on observed pixels only: the invalid wedges are
            # stored as NaN here and would otherwise read as strong contrast.
            observed = hr_win[np.isfinite(hr_win)]
            if observed.size == 0:
                continue
            if float(observed.std()) < spec.min_std and rng.random() >= spec.max_flat_fraction:
                continue
            yield {
                "lr": lr_win,
                "hr": np.nan_to_num(hr_win, nan=0.0),
                "mask": hr_win_valid,
            }


def _flush_pairs(split_dir: Path, shard_no: int, buffers: list[dict],
                 sites: list[str], tiles: list[str]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "lr": np.stack([_to_uint16(b["lr"]) for b in buffers]),
        "hr": np.stack([_to_uint16(b["hr"]) for b in buffers]),
        "mask": np.stack([b["mask"].astype(np.uint8) for b in buffers]),
        # Carried through so evaluation can report per geographic area, which
        # is where SR failures actually concentrate. A single mean over a test
        # set hides a model that fails on every built-up tile.
        "site": np.array(sites, dtype="U8"),
        "tile_id": np.array(tiles, dtype="U40"),
    }
    np.savez_compressed(split_dir / f"shard_{shard_no:05d}.npz", **payload)
    buffers.clear()
    sites.clear()
    tiles.clear()


def build_pair_patches(
    splits: dict[str, Sequence[TilePair]],
    root: str | Path,
    out_dir: str | Path,
    spec: PairPatchSpec | None = None,
    bands: Sequence[str] = RGB_NIR_BANDS,
    seed: int = 1337,
    logger=None,
) -> dict:
    """Write train/val/test shards of real LR/HR/mask triples.

    Shards carry the HR *valid mask* alongside the imagery. Without it the
    loss would be computed over NEON's empty flight-line wedges, which are
    stored as zeros and would teach the model to darken image borders.
    """
    spec = spec or PairPatchSpec()
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    counts: dict[str, int] = {}
    per_split_sites: dict[str, list[str]] = {}
    failures: list[dict[str, str]] = []

    for split, pairs in splits.items():
        buffers: list[dict[str, np.ndarray]] = []
        sites: list[str] = []
        tiles: list[str] = []
        shard_no = 0
        kept = 0
        for pair in pairs:
            try:
                data = read_pair(pair, root, bands, spec.scale)
            except (ValueError, OSError) as exc:
                failures.append({"tile": pair.tile_id, "error": str(exc)})
                if logger:
                    logger.warning("skipping %s: %s", pair.tile_id, exc)
                continue
            for patch in iter_pair_patches(data, spec, rng):
                buffers.append(patch)
                sites.append(pair.site)
                tiles.append(pair.tile_id)
                kept += 1
                if len(buffers) >= spec.shard_size:
                    _flush_pairs(out_dir / split, shard_no, buffers, sites, tiles)
                    shard_no += 1
        if buffers:
            _flush_pairs(out_dir / split, shard_no, buffers, sites, tiles)
        counts[split] = kept
        per_split_sites[split] = sorted({p.site for p in pairs})
        if logger:
            logger.info("%s: %d patches from %d tiles across %d sites",
                        split, kept, len(pairs), len(per_split_sites[split]))

    manifest = {
        "source": "SEN2NEON",
        "repo_id": REPO_ID,
        "hr_product": HR_DIR,
        "lr_product": LR_DIR,
        "pairs_kind": "native",       # real observations, not a simulated LR
        "bands": [normalise_band(b) for b in bands],
        "band_indices_1based": list(band_indices(bands)),
        "scale": spec.scale,
        "lr_patch": spec.lr_patch,
        "hr_patch": spec.hr_patch,
        "patch_spec": asdict(spec),
        "counts": counts,
        "sites": per_split_sites,
        "reflectance_scale": S2_REFLECTANCE_SCALE,
        "normalization": "reflectance = DN / 10000, fixed; no statistics fitted from data",
        "storage": "uint16 hr/lr scaled by reflectance_scale, uint8 hr validity mask",
        "has_mask": True,
        "failures": failures,
        "seed": seed,
        "limitations": [
            "HR reference is NEON AVIRIS-NG convolved to Sentinel-2 SRFs, not a "
            "Sentinel-2 observation; cross-sensor radiometric differences remain.",
            "LR and HR acquisitions are separated by up to 40 days, so part of "
            "any residual is real surface change rather than model error.",
            "Cloud information is a per-tile Cloud Score+ CDF; the 12-band stack "
            "carries no SCL layer, so no per-pixel cloud mask is applied.",
            "Coverage is North American NEON sites, imbalanced toward natural "
            "land cover; metrics do not transfer automatically to other regions.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _to_uint16(patch: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(patch) * S2_REFLECTANCE_SCALE, 0, 65535).astype(np.uint16)
