"""Input inspection and admission control for GeoTIFF scenes.

Two jobs, and they are deliberately in one place so the interface and the CLI
cannot drift apart on what counts as an acceptable input:

1. **Describe** a raster -- dimensions, band count, CRS, pixel spacing, valid
   coverage, dynamic range. This is what the dashboard's input summary shows.
2. **Admit or reject** it against a checkpoint, with a reason a user can act
   on. "Rejected" is useless; "this file has 12 bands and the model expects
   4 -- choose which four in the band mapping panel" is not.

The distinction between an *error* and a *warning* here is whether the
product would be wrong or merely caveated. No CRS means the output cannot be
placed on a map, so it is an error. A 20 m pixel spacing when the model was
trained at 10 m still produces an image, but it is a 4x step from the wrong
starting scale, so it is a loud warning and the report says why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Sentinel-2 surface reflectance is nominally 0-1 after scaling. Values far
#: outside this usually mean the file is raw DN that was not recognised as
#: integer, or radiance rather than reflectance.
PLAUSIBLE_REFLECTANCE_MAX = 2.0

#: The pixel spacing this project's models are trained for, in metres.
EXPECTED_INPUT_GSD = 10.0
GSD_TOLERANCE = 0.5


@dataclass
class InputReport:
    """Everything known about an input raster before any model touches it."""

    path: str
    width: int
    height: int
    band_count: int
    band_names: list[str]
    dtype: str
    crs: str | None
    epsg: int | None
    transform: Any
    pixel_size_x: float | None
    pixel_size_y: float | None
    bounds: tuple[float, float, float, float] | None
    nodata: float | None
    valid_fraction: float
    value_min: float
    value_max: float
    is_integer_source: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the file can be processed into a correct georeferenced product."""
        return not self.errors

    @property
    def pixel_size(self) -> float | None:
        return self.pixel_size_x

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def output_shape(self, scale: int) -> tuple[int, int]:
        return self.height * scale, self.width * scale

    def output_gsd(self, scale: int) -> float | None:
        return (self.pixel_size_x / scale) if self.pixel_size_x else None

    def ground_extent_m(self) -> tuple[float, float] | None:
        """Width and height on the ground, which super-resolution must preserve."""
        if not (self.pixel_size_x and self.pixel_size_y):
            return None
        return self.width * self.pixel_size_x, self.height * abs(self.pixel_size_y)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "dimensions": f"{self.width} x {self.height}",
            "bands": self.band_count,
            "band_names": self.band_names,
            "dtype": self.dtype,
            "crs": self.crs,
            "epsg": self.epsg,
            "pixel_size_m": self.pixel_size_x,
            "bounds": list(self.bounds) if self.bounds else None,
            "nodata": self.nodata,
            "valid_fraction": round(self.valid_fraction, 6),
            "value_range": [self.value_min, self.value_max],
            "errors": self.errors,
            "warnings": self.warnings,
        }


def inspect_raster(path: str | Path, sample_limit: int = 4_000_000) -> InputReport:
    """Read a raster's geometry and a value sample, without loading it whole.

    Large scenes are sampled with a decimated read for the coverage and range
    statistics: a 10980x10980 Sentinel-2 tile is 1.9 GB as float32 and the
    summary panel does not need every pixel to report that 3% is nodata.
    """
    import rasterio
    from rasterio.errors import RasterioIOError

    path = Path(path)
    try:
        src = rasterio.open(str(path))
    except RasterioIOError as exc:
        return _unreadable(path, f"rasterio cannot open this file: {exc}")

    with src:
        width, height, count = src.width, src.height, src.count
        descriptions = src.descriptions or ()
        band_names = [descriptions[i] or f"band_{i + 1}" for i in range(count)]
        dtype = str(src.dtypes[0])
        is_integer = bool(np.issubdtype(np.dtype(dtype), np.integer))
        crs = src.crs
        transform = src.transform
        nodata = src.nodata

        # Decimated read: at most `sample_limit` pixels per band.
        step = max(1, int(np.sqrt((width * height * max(count, 1)) / sample_limit)))
        out_shape = (count, max(1, height // step), max(1, width // step))
        sample = src.read(out_shape=out_shape).astype(np.float32)

    finite = np.isfinite(sample)
    if nodata is not None:
        finite &= sample != float(nodata)
    valid = finite.all(axis=0)
    observed = sample[finite]
    scaled = observed / 10000.0 if is_integer else observed

    report = InputReport(
        path=str(path),
        width=width,
        height=height,
        band_count=count,
        band_names=band_names,
        dtype=dtype,
        crs=str(crs) if crs else None,
        epsg=(crs.to_epsg() if crs else None),
        transform=transform,
        pixel_size_x=(abs(transform.a) if transform else None),
        pixel_size_y=(abs(transform.e) if transform else None),
        bounds=(tuple(rasterio.transform.array_bounds(height, width, transform))
                if transform else None),
        nodata=(float(nodata) if nodata is not None else None),
        valid_fraction=float(valid.mean()) if valid.size else 0.0,
        value_min=float(scaled.min()) if scaled.size else float("nan"),
        value_max=float(scaled.max()) if scaled.size else float("nan"),
        is_integer_source=is_integer,
    )
    _apply_rules(report)
    return report


def _unreadable(path: Path, reason: str) -> InputReport:
    return InputReport(
        path=str(path), width=0, height=0, band_count=0, band_names=[], dtype="?",
        crs=None, epsg=None, transform=None, pixel_size_x=None, pixel_size_y=None,
        bounds=None, nodata=None, valid_fraction=0.0, value_min=float("nan"),
        value_max=float("nan"), is_integer_source=False,
        errors=[f"{reason} Supply a GeoTIFF readable by GDAL."],
    )


def _apply_rules(r: InputReport) -> None:
    """Populate errors and warnings. Every message says what to do about it."""
    if r.width < 8 or r.height < 8:
        r.errors.append(
            f"Image is {r.width}x{r.height} pixels, too small to tile. "
            "Supply a scene of at least 8x8 pixels."
        )
    if r.band_count == 0:
        r.errors.append("The file contains no raster bands.")

    if r.crs is None:
        r.errors.append(
            "No coordinate reference system. The output could be written but not "
            "placed on a map, so georeferenced export is refused. Assign a CRS "
            "first, e.g. `gdal_edit.py -a_srs EPSG:32643 scene.tif`."
        )
    elif r.epsg is None:
        r.warnings.append(
            "The CRS has no EPSG code. The output will carry the same CRS "
            "definition, but some GIS tools handle unlabelled CRSs poorly."
        )

    if r.transform is None:
        r.errors.append(
            "No affine transform (geotransform). Pixel spacing and bounds are "
            "unknown, so the 4x output grid cannot be positioned."
        )
    else:
        if r.pixel_size_x and r.pixel_size_y:
            if abs(r.pixel_size_x - r.pixel_size_y) > 1e-6 * max(r.pixel_size_x, 1.0):
                r.warnings.append(
                    f"Non-square pixels ({r.pixel_size_x:.4g} x {r.pixel_size_y:.4g}). "
                    "Both axes are divided by the scale factor, so the output "
                    "inherits the same anisotropy."
                )
            gsd = r.pixel_size_x
            if abs(gsd - EXPECTED_INPUT_GSD) > GSD_TOLERANCE:
                r.warnings.append(
                    f"Pixel spacing is {gsd:.4g} m, not the {EXPECTED_INPUT_GSD:.0f} m "
                    "the models were trained at. The network will still apply a 4x "
                    f"step, producing a {gsd / 4:.4g} m grid, but it is operating "
                    "outside the scale range it learned and the result is not "
                    "comparable to the reported metrics."
                )

    if r.valid_fraction <= 0.0:
        r.errors.append(
            "Every sampled pixel is nodata or non-finite. Check the nodata value "
            "declared in the file."
        )
    elif r.valid_fraction < 0.5:
        r.warnings.append(
            f"Only {r.valid_fraction:.1%} of pixels carry data. Invalid pixels are "
            "masked and returned as nodata, and metrics are computed over the "
            "valid area only."
        )

    if np.isfinite(r.value_max) and r.value_max > PLAUSIBLE_REFLECTANCE_MAX:
        r.warnings.append(
            f"Values reach {r.value_max:.3g} after scaling, above the plausible "
            "surface-reflectance range. If this file is radiance or an unscaled "
            "product, the normalisation assumed here (DN / 10000) is wrong and "
            "the output will be radiometrically meaningless."
        )
    if np.isfinite(r.value_min) and r.value_min < -0.05:
        r.warnings.append(
            f"Minimum value is {r.value_min:.3g}. Negative reflectance usually "
            "means an unmasked nodata sentinel."
        )


#: Band layouts for products that carry no band descriptions, keyed by band
#: count. SEN2NEON GeoTIFFs are written with `descriptions == (None,) * 12`,
#: so without this the name lookup finds nothing and falls straight through to
#: positional selection -- the exact failure the name lookup exists to prevent.
#: These are documented layouts, not guesses, and every use says so.
KNOWN_LAYOUTS: dict[int, tuple[str, tuple[str, ...]]] = {
    12: ("Sentinel-2 12-band stack (SEN2NEON; B10 absent)",
         ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")),
    13: ("Sentinel-2 L1C 13-band stack",
         ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")),
    4:  ("this project's 4-band product",
         ("B02", "B03", "B04", "B08")),
}


def infer_band_layout(report: InputReport) -> tuple[list[str], str | None]:
    """Band names for `report`, from its descriptions or a documented layout.

    Returns ``(names, note)``. `note` is None when the file named its own
    bands, and otherwise states which layout was assumed -- an assumption the
    caller is expected to show the user, because it is the one place a wrong
    guess is both plausible and silent.
    """
    described = [n for n in report.band_names if n and not n.startswith("band_")]
    if len(described) == report.band_count and report.band_count:
        return list(report.band_names), None

    layout = KNOWN_LAYOUTS.get(report.band_count)
    if layout is None:
        return list(report.band_names), (
            f"This file has {report.band_count} unnamed bands and no documented "
            "layout matches that count. Bands are selected by position; set the "
            "mapping explicitly if that is wrong."
        )
    description, names = layout
    return list(names), (
        f"This file names no bands. Assuming the {description}: "
        + ", ".join(f"{i}={n}" for i, n in enumerate(names, start=1))
        + ". Override in the band mapping if the file differs."
    )


def resolve_band_mapping(
    report: InputReport,
    requested: Sequence[str] | None,
    expected_count: int = 4,
) -> tuple[list[int], list[str], list[str]]:
    """Choose which file bands feed the model, and explain the choice.

    Returns ``(1-based indices, resolved names, notes)``. Bands are matched by
    name against the file's own descriptions, or against a documented layout
    for the band count when the file names nothing. Positional selection is
    the last resort and always says so, because silently taking the first four
    bands of a 12-band Sentinel-2 stack yields coastal-aerosol/blue/green/red
    in place of blue/green/red/NIR.
    """
    notes: list[str] = []
    available, layout_note = infer_band_layout(report)
    if layout_note:
        notes.append(layout_note)

    if requested:
        from .data.sen2neon import normalise_band

        lookup: dict[str, int] = {}
        for i, name in enumerate(available, start=1):
            try:
                lookup.setdefault(normalise_band(name), i)
            except ValueError:
                continue
        indices: list[int] = []
        for want in requested:
            key = normalise_band(want)
            if key in lookup:
                indices.append(lookup[key])
            else:
                indices = []
                break
        if indices:
            source = "the assumed layout" if layout_note else "the file's band descriptions"
            notes.append(
                f"Bands matched by name against {source}: "
                + ", ".join(f"{n}=band {i}" for n, i in zip(requested, indices))
            )
            return indices, [normalise_band(b) for b in requested], notes

    if report.band_count < expected_count:
        raise ValueError(
            f"The model needs {expected_count} bands but this file has "
            f"{report.band_count}. Supply a {expected_count}-band raster, or "
            "a checkpoint trained on this many bands."
        )
    if report.band_count > expected_count:
        notes.append(
            f"**Falling back to position.** This file has {report.band_count} "
            f"bands, the model takes {expected_count}, and the requested band "
            "names could not be matched. The first "
            f"{expected_count} are used, which for a Sentinel-2 stack is very "
            "likely wrong -- set the band mapping explicitly."
        )
    else:
        notes.append(f"All {expected_count} bands used in file order.")
    indices = list(range(1, expected_count + 1))
    return indices, [available[i - 1] for i in indices], notes


def check_against_checkpoint(report: InputReport, info: dict) -> list[str]:
    """Blocking reasons this checkpoint cannot process this input."""
    from .models.registry import compatibility_report

    problems = compatibility_report(info, n_bands=len(info.get("bands") or []) or 4)
    # Drop the band-count check here: band *mapping* resolves that, and this
    # function runs before the user has chosen one.
    return [p for p in problems if "bands" not in p]
