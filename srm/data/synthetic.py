"""Procedural high-resolution scenes, so the pipeline runs with no downloads.

These are not a substitute for real imagery -- no paper result should rest on
them. They exist so the repository is runnable end-to-end on a fresh machine
without Copernicus credentials, and so the unit tests have deterministic input.

The generator is built to exercise exactly the structures that matter for SRM
and that 10 m pixels destroy: field boundaries, roads 2.5-5 m wide, individual
building footprints, and water edges. Each class carries a plausible 4-band
reflectance signature, so NDVI/NDWI computed on these scenes behaves sensibly
and the spectral losses have something real to preserve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Surface reflectance endmembers for B02, B03, B04, B08 (0-1).
# Values follow typical Sentinel-2 L2A spectra for each cover type.
ENDMEMBERS: dict[str, tuple[float, float, float, float]] = {
    "water":       (0.030, 0.045, 0.030, 0.012),
    "turbid":      (0.055, 0.080, 0.070, 0.045),
    "veg_dense":   (0.028, 0.060, 0.032, 0.430),
    "veg_mid":     (0.040, 0.075, 0.055, 0.330),
    "veg_sparse":  (0.060, 0.095, 0.100, 0.250),
    "soil_dry":    (0.120, 0.160, 0.205, 0.275),
    "soil_wet":    (0.080, 0.100, 0.125, 0.180),
    "concrete":    (0.175, 0.190, 0.200, 0.225),
    "roof_bright": (0.230, 0.245, 0.260, 0.285),
    "roof_dark":   (0.070, 0.075, 0.080, 0.100),
    "asphalt":     (0.085, 0.095, 0.105, 0.120),
}
CROP_CLASSES = ("veg_dense", "veg_mid", "veg_sparse", "soil_dry", "soil_wet")

# Integer label map written alongside each scene, for the land-cover application.
LABELS = {"other": 0, "water": 1, "vegetation": 2, "soil": 3, "built": 4, "road": 5}


@dataclass
class SceneSpec:
    size: int = 1024
    gsd: float = 2.5             # metres per pixel of the generated HR scene
    n_fields: int = 46
    n_urban_blocks: int = 5
    road_density: float = 1.0
    river: bool = True
    haze: float = 0.004          # additive atmospheric path radiance
    texture: float = 0.55        # 0 = flat classes, 1 = strong within-parcel texture


def _cv2():
    import cv2

    return cv2


def _fractal_noise(size: int, rng: np.random.Generator, octaves: int = 5,
                   persistence: float = 0.55) -> np.ndarray:
    """Value-noise sum of octaves in [0, 1]; stands in for soil/canopy variation."""
    cv2 = _cv2()
    total = np.zeros((size, size), dtype=np.float32)
    amplitude, norm = 1.0, 0.0
    for octave in range(octaves):
        res = max(2, int(size / (2 ** (octaves - octave))))
        grid = rng.random((res, res)).astype(np.float32)
        layer = cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        total += amplitude * layer
        norm += amplitude
        amplitude *= persistence
    total /= max(norm, 1e-6)
    return np.clip(total, 0.0, 1.0)


def _agricultural_parcels(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    """Parcel id map from a jittered, globally rotated grid of rectangles."""
    cv2 = _cv2()
    size = spec.size
    parcels = np.zeros((size, size), dtype=np.int32)
    cols = max(1, int(np.sqrt(spec.n_fields)))
    rows = max(1, int(np.ceil(spec.n_fields / cols)))
    theta = rng.uniform(0, np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = np.array([size / 2, size / 2])
    cell_w, cell_h = size * 1.6 / cols, size * 1.6 / rows

    pid = 1
    for r in range(rows):
        for c in range(cols):
            x0 = (c - cols / 2) * cell_w + rng.uniform(-0.12, 0.12) * cell_w
            y0 = (r - rows / 2) * cell_h + rng.uniform(-0.12, 0.12) * cell_h
            w = cell_w * rng.uniform(0.72, 0.99)
            h = cell_h * rng.uniform(0.72, 0.99)
            corners = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
            pts = (corners @ rot.T) + centre
            cv2.fillPoly(parcels, [pts.astype(np.int32)], int(pid))
            pid += 1
    return parcels


def _road_network(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    """Binary road mask with widths of 1-2 px, i.e. 2.5-5 m of real road.

    This is the feature that motivates the whole project: at 10 m these roads
    are a sub-pixel signal, and recovering them is the headline capability.
    """
    cv2 = _cv2()
    size = spec.size
    mask = np.zeros((size, size), dtype=np.uint8)
    n_trunk = max(1, int(3 * spec.road_density))
    for _ in range(n_trunk):
        horizontal = rng.random() < 0.5
        offset = int(rng.uniform(0.15, 0.85) * size)
        drift = rng.integers(-size // 12, size // 12)
        p0 = (0, offset) if horizontal else (offset, 0)
        p1 = (size, offset + drift) if horizontal else (offset + drift, size)
        cv2.line(mask, p0, p1, 1, thickness=int(rng.integers(2, 4)), lineType=cv2.LINE_AA)
    n_minor = int(9 * spec.road_density)
    for _ in range(n_minor):
        p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        length = rng.integers(size // 5, size // 2)
        angle = rng.uniform(0, 2 * np.pi)
        p1 = (int(p0[0] + length * np.cos(angle)), int(p0[1] + length * np.sin(angle)))
        cv2.line(mask, p0, p1, 1, thickness=1, lineType=cv2.LINE_AA)
    return mask


def _urban_blocks(spec: SceneSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Building footprint mask plus a per-building brightness class."""
    cv2 = _cv2()
    size = spec.size
    built = np.zeros((size, size), dtype=np.uint8)
    bright = np.zeros((size, size), dtype=np.uint8)
    for _ in range(spec.n_urban_blocks):
        bx = int(rng.integers(0, max(1, size - size // 5)))
        by = int(rng.integers(0, max(1, size - size // 5)))
        bw = int(rng.integers(size // 10, size // 4))
        bh = int(rng.integers(size // 10, size // 4))
        n_buildings = int(rng.integers(12, 45))
        for _ in range(n_buildings):
            # Footprints of 5-20 m across: several HR pixels, a fraction of an LR one.
            w = int(rng.integers(2, 9))
            h = int(rng.integers(2, 9))
            x = int(rng.integers(bx, max(bx + 1, bx + bw - w)))
            y = int(rng.integers(by, max(by + 1, by + bh - h)))
            if x + w >= size or y + h >= size:
                continue
            cv2.rectangle(built, (x, y), (x + w, y + h), 1, -1)
            if rng.random() < 0.45:
                cv2.rectangle(bright, (x, y), (x + w, y + h), 1, -1)
    return built, bright


def _river(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    cv2 = _cv2()
    size = spec.size
    mask = np.zeros((size, size), dtype=np.uint8)
    n_pts = 7
    xs = np.linspace(-10, size + 10, n_pts)
    ys = np.cumsum(rng.normal(0, size * 0.09, n_pts)) + rng.uniform(0.25, 0.75) * size
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    width = int(rng.integers(3, 10))
    cv2.polylines(mask, [pts], False, 1, thickness=width, lineType=cv2.LINE_AA)
    if rng.random() < 0.5:  # a narrow tributary, a genuinely sub-10 m feature
        t_pts = pts[::2] + np.array([0, int(rng.integers(40, 140))])
        cv2.polylines(mask, [t_pts], False, 1, thickness=2, lineType=cv2.LINE_AA)
    return mask


def generate_scene(
    spec: SceneSpec | None = None,
    seed: int | None = None,
    bands: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reflectance (B, H, W) float32, label map (H, W) uint8)."""
    spec = spec or SceneSpec()
    rng = np.random.default_rng(seed)
    size = spec.size

    parcels = _agricultural_parcels(spec, rng)
    roads = _road_network(spec, rng)
    built, bright = _urban_blocks(spec, rng)
    water = _river(spec, rng) if spec.river else np.zeros((size, size), np.uint8)
    texture = _fractal_noise(size, rng)

    image = np.zeros((4, size, size), dtype=np.float32)
    labels = np.zeros((size, size), dtype=np.uint8)

    # 1. Agricultural / natural background, one spectral class per parcel.
    for pid in np.unique(parcels):
        mask = parcels == pid
        if not mask.any():
            continue
        cls = CROP_CLASSES[int(rng.integers(0, len(CROP_CLASSES)))] if pid > 0 else "veg_sparse"
        base = np.array(ENDMEMBERS[cls], dtype=np.float32)
        # Within-parcel variation: real fields are never uniform, and this is
        # the texture the model has to learn to synthesise rather than invent.
        amp = spec.texture * rng.uniform(0.08, 0.22)
        modulation = 1.0 + amp * (texture[mask] - 0.5) * 2.0
        image[:, mask] = base[:, None] * modulation[None, :]
        labels[mask] = LABELS["soil"] if cls.startswith("soil") else LABELS["vegetation"]

    # 2. Water, 3. buildings, 4. roads -- painted in order of who wins overlaps.
    if water.any():
        m = water.astype(bool)
        cls = "turbid" if rng.random() < 0.35 else "water"
        base = np.array(ENDMEMBERS[cls], dtype=np.float32)
        image[:, m] = base[:, None] * (1.0 + 0.10 * (texture[m] - 0.5)[None, :])
        labels[m] = LABELS["water"]

    for mask, cls in ((built.astype(bool) & ~bright.astype(bool), "roof_dark"),
                      (bright.astype(bool), "roof_bright")):
        if mask.any():
            base = np.array(ENDMEMBERS[cls], dtype=np.float32)
            image[:, mask] = base[:, None] * (1.0 + 0.12 * (texture[mask] - 0.5)[None, :])
            labels[mask] = LABELS["built"]

    road_mask = roads.astype(bool) & ~water.astype(bool)
    if road_mask.any():
        base = np.array(ENDMEMBERS["asphalt"], dtype=np.float32)
        image[:, road_mask] = base[:, None] * (1.0 + 0.08 * (texture[road_mask] - 0.5)[None, :])
        labels[road_mask] = LABELS["road"]

    # Atmospheric haze: strongest in the blue band, as Rayleigh scattering is.
    haze = np.array([1.0, 0.70, 0.45, 0.12], dtype=np.float32) * spec.haze
    image += haze[:, None, None]
    image = np.clip(image, 0.0, 1.0)

    if bands != 4:
        image = image[:bands] if bands < 4 else np.concatenate(
            [image, np.repeat(image[-1:], bands - 4, axis=0)], axis=0
        )
    return image.astype(np.float32), labels


def generate_bitemporal(
    spec: SceneSpec | None = None,
    seed: int | None = None,
    change_fraction: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two dates over the same ground plus a binary change mask.

    Changes are deliberately small -- new building rows, a harvested field, a
    flooded patch -- because the question for SRM is whether super-resolution
    recovers changes that are sub-pixel at 10 m.
    """
    cv2 = _cv2()
    spec = spec or SceneSpec()
    rng = np.random.default_rng(seed)
    t1, labels = generate_scene(spec, seed=seed)
    t2 = t1.copy()
    size = spec.size
    change = np.zeros((size, size), dtype=np.uint8)

    n_events = max(3, int(change_fraction * 150))
    for _ in range(n_events):
        kind = rng.choice(["build", "harvest", "flood"], p=[0.45, 0.35, 0.20])
        patch = np.zeros((size, size), dtype=np.uint8)
        x, y = int(rng.integers(0, size - 60)), int(rng.integers(0, size - 60))
        if kind == "build":
            for _ in range(int(rng.integers(3, 12))):
                w, h = int(rng.integers(2, 7)), int(rng.integers(2, 7))
                bx, by = x + int(rng.integers(0, 50)), y + int(rng.integers(0, 50))
                cv2.rectangle(patch, (bx, by), (min(bx + w, size - 1), min(by + h, size - 1)), 1, -1)
            target = ENDMEMBERS["roof_bright"]
        elif kind == "harvest":
            w, h = int(rng.integers(40, 130)), int(rng.integers(40, 130))
            cv2.rectangle(patch, (x, y), (min(x + w, size - 1), min(y + h, size - 1)), 1, -1)
            target = ENDMEMBERS["soil_dry"]
        else:
            cv2.circle(patch, (x + 30, y + 30), int(rng.integers(12, 45)), 1, -1)
            target = ENDMEMBERS["turbid"]
        mask = patch.astype(bool)
        if not mask.any():
            continue
        base = np.array(target, dtype=np.float32)
        new = base[:, None] * (1.0 + 0.10 * rng.normal(0, 1, mask.sum())[None, :] * 0.2)
        # Blend rather than replace. A real change rarely fills its pixels
        # completely, and a full-contrast replacement would make the change
        # detectable even at 10 m, removing the headroom the SRM claim rests on.
        alpha = float(rng.uniform(0.45, 0.85))
        t2[:, mask] = alpha * new + (1.0 - alpha) * t2[:, mask]
        change[mask] = 1

    # A mild global radiometric shift between dates, as different sun angles give.
    t2 = np.clip(t2 * float(rng.uniform(0.96, 1.05)), 0.0, 1.0)
    return t1.astype(np.float32), t2.astype(np.float32), change


def write_scene_geotiff(
    out_path: str | Path,
    image: np.ndarray,
    gsd: float = 2.5,
    origin: tuple[float, float] = (500000.0, 4500000.0),
    epsg: int = 32643,
    band_names: tuple[str, ...] = ("B02", "B03", "B04", "B08"),
):
    """Write a synthetic scene as a real, georeferenced GeoTIFF (UTM by default).

    Giving synthetic data a genuine CRS means the geospatial code paths are
    exercised by the demo, not only by real downloads.
    """
    from affine import Affine
    from rasterio.crs import CRS

    from ..utils.geo import GeoMeta, write_raster

    meta = GeoMeta(
        crs=CRS.from_epsg(epsg),
        transform=Affine(gsd, 0.0, origin[0], 0.0, -gsd, origin[1]),
        band_names=list(band_names[: image.shape[0]]),
    )
    return write_raster(out_path, image, meta, scale_factor=1, dtype="float32")
