"""Does this checkout work on this machine? Run this first.

    python scripts/check_install.py

Checks the things that actually differ between machines, in the order they
would bite: interpreter version, the packages whose wheels are platform
specific, whether a GPU was found, whether any trained checkpoint is
present and loadable, and finally a real end-to-end super-resolution of a
bundled scene with its georeferencing verified.

Exits non-zero if anything required is missing, so it is usable in CI. A
missing GPU or a missing training dataset is reported but not fatal: neither
is needed to run the dashboard.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "

#: (import name, pip name, minimum). Only packages whose absence actually
#: stops something; the rest surface naturally.
REQUIRED = [
    ("numpy", "numpy", (1, 26)),
    ("torch", "torch", (2, 5)),
    ("rasterio", "rasterio", (1, 3)),
    ("affine", "affine", None),
    ("yaml", "PyYAML", None),
    ("skimage", "scikit-image", None),
    ("matplotlib", "matplotlib", None),
    ("PIL", "Pillow", None),
    ("streamlit", "streamlit", (1, 32)),
    ("pandas", "pandas", None),
]
OPTIONAL = [
    ("huggingface_hub", "huggingface_hub", "downloading SEN2NEON"),
    ("streamlit_image_comparison", "streamlit-image-comparison", "before/after slider"),
    ("cv2", "opencv-python-headless", "fast resampling in the degradation model"),
]

failures: list[str] = []
warnings: list[str] = []


def line(status: str, text: str) -> None:
    print(f"[{status}] {text}")


def _version(module) -> str:
    return str(getattr(module, "__version__", "?"))


def _tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in text.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_python() -> None:
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro} ({platform.machine()}, {sys.platform})"
    if v < (3, 10):
        failures.append("Python 3.10 or newer is required")
        line(BAD, text + " -- needs >= 3.10")
    else:
        line(OK, text)


def check_packages() -> None:
    for name, pip_name, minimum in REQUIRED:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failures.append(f"pip install {pip_name}   ({type(exc).__name__})")
            line(BAD, f"{pip_name} missing")
            continue
        version = _version(module)
        if minimum and _tuple(version) and _tuple(version) < minimum:
            warnings.append(f"{pip_name} {version} is older than "
                            f"{'.'.join(map(str, minimum))}")
            line(WARN, f"{pip_name} {version} (older than recommended)")
        else:
            line(OK, f"{pip_name} {version}")

    for name, pip_name, why in OPTIONAL:
        try:
            module = importlib.import_module(name)
            line(OK, f"{pip_name} {_version(module)}")
        except Exception:
            warnings.append(f"{pip_name} absent -- {why} unavailable")
            line(WARN, f"{pip_name} absent ({why})")


def check_device() -> None:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        line(OK, f"CUDA available: {name} ({total:.1f} GB)")
    else:
        line(WARN, "no CUDA GPU -- everything runs on CPU, a few seconds per tile")
        warnings.append("no GPU: inference is CPU-bound but fully functional")


def check_checkpoints() -> list[dict]:
    from srm.service import discover_checkpoints, trained_checkpoints

    found = discover_checkpoints([ROOT / "checkpoints"])
    usable = trained_checkpoints(found)
    if not usable:
        failures.append(
            "no trained checkpoint under checkpoints/ -- the dashboard will "
            "refuse AI enhancement. Copy one over, or train with "
            "`python -m srm.train --config configs/sen2neon_x4.yaml`"
        )
        line(BAD, f"{len(found)} checkpoint file(s), none trained")
        return []
    line(OK, f"{len(usable)} trained checkpoint(s):")
    for info in usable:
        print(f"         {info['name']:<30} {info['architecture']} x{info['scale']}  "
              f"epoch {info['epoch']:<4} {info['weights']} weights  "
              f"bands {','.join(info['bands'])}")
    return usable


def check_samples() -> list[Path]:
    scenes = sorted((ROOT / "data" / "samples").glob("*_s2_10m.tif"))
    if not scenes:
        warnings.append("no bundled sample scenes under data/samples/")
        line(WARN, "no sample scenes -- upload your own GeoTIFF in the dashboard")
    else:
        line(OK, f"{len(scenes)} sample scene(s) with references")
    return scenes


def check_training_data() -> None:
    manifest = ROOT / "data" / "patches_sen2neon" / "manifest.json"
    if manifest.is_file():
        import json

        counts = json.loads(manifest.read_text(encoding="utf-8")).get("counts", {})
        line(OK, f"training patches present: {counts}")
    else:
        line(WARN, "no training dataset -- needed only to train, not to run")
        warnings.append(
            "to train: python scripts/download_sen2neon.py --limit 150 "
            "&& python scripts/prepare_sen2neon.py"
        )


def check_inference(usable: list[dict], scenes: list[Path]) -> None:
    """The real test: super-resolve a scene and verify the geometry survived."""
    if not usable or not scenes:
        line(WARN, "skipping the end-to-end check (needs a checkpoint and a scene)")
        return

    import numpy as np
    import rasterio

    from srm import service

    scene, info = str(scenes[0]), usable[0]
    report = service.describe_input(scene)
    if not report.ok:
        failures.append(f"sample scene rejected: {report.errors}")
        line(BAD, "sample scene failed validation")
        return
    plan = service.plan_bands(report, info)
    result = service.run_super_resolution(scene, info, plan["indices"],
                                          device="cpu", tile=128, overlap=16)
    out = ROOT / "outputs" / "install_check"
    written = service.write_outputs(result, out)

    with rasterio.open(scene) as a, rasterio.open(written["enhanced_geotiff"]) as b:
        same_crs = a.crs == b.crs
        same_ground = np.allclose(a.bounds, b.bounds, atol=1e-6)
        upscaled = (b.width, b.height) == (a.width * result.scale,
                                           a.height * result.scale)
    if same_crs and same_ground and upscaled:
        line(OK, f"end-to-end: {a.width}x{a.height} @ {report.pixel_size} m -> "
                 f"{b.width}x{b.height} @ {result.output_gsd} m in "
                 f"{result.seconds_model:.1f}s, CRS and ground extent preserved")
        print(f"         wrote {Path(written['enhanced_geotiff']).relative_to(ROOT)}")
    else:
        failures.append("the output is not georeferenced correctly")
        line(BAD, f"geometry check failed (crs={same_crs} bounds={same_ground} "
                  f"size={upscaled})")


def main(argv=None) -> None:
    # argv is accepted (and ignored) so `srm check` can delegate here the
    # same way it does for every other script in this directory.
    print(f"\nSRM install check -- {ROOT}\n" + "-" * 62)
    check_python()
    check_packages()
    if failures:
        _summary()
        return
    check_device()
    usable = check_checkpoints()
    scenes = check_samples()
    check_training_data()
    check_inference(usable, scenes)
    _summary()


def _summary() -> None:
    print("-" * 62)
    if failures:
        print(f"\n{len(failures)} problem(s) must be fixed:\n")
        for f in failures:
            print(f"  - {f}")
    if warnings:
        print(f"\n{len(warnings)} note(s):\n")
        for w in warnings:
            print(f"  - {w}")
    if not failures:
        print("\nReady. Start the dashboard with:\n\n    streamlit run app/app.py\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
