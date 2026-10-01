"""The dashboard script actually executes.

`py_compile` and a linter cannot catch a runtime TypeError, and neither can
any test of `srm.service`, because the view is where the wiring lives. This
suite ran green while `app.py` crashed on every single page load:

    by_path = {c["path"]: c for c in usable}
    ...
    [c["path"] for c in by_path]        # iterates KEYS -- strings

`st.testing.v1.AppTest` runs the script in-process with no browser, so the
whole view is exercised in about a second.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS

from srm.models.registry import build_model, save_checkpoint

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "app" / "app.py"

LAYOUT = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
FOREST = {"B1": 0.11, "B2": 0.029, "B3": 0.054, "B4": 0.026, "B5": 0.20, "B6": 0.35,
          "B7": 0.40, "B8": 0.464, "B8A": 0.44, "B9": 0.30, "B11": 0.18, "B12": 0.08}


def _scene(path: Path, size: int = 32, pixel: float = 10.0) -> str:
    rng = np.random.default_rng(0)
    stack = np.stack([np.full((size, size), FOREST[b], dtype=np.float32)
                      + rng.normal(0, 0.002, (size, size)).astype(np.float32)
                      for b in LAYOUT])
    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 12,
        "dtype": "uint16", "crs": CRS.from_epsg(32617),
        "transform": Affine(pixel, 0, 539120.0, 0, -pixel, 4146000.0),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.clip(stack * 10000, 0, 65535).astype(np.uint16))
    return str(path)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A checkpoints/ and data/ tree the app will discover, isolated from the repo."""
    cfg = {"experiment": "t", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    torch.manual_seed(0)
    for name, seed in (("run_a", 1), ("run_b", 2)):
        d = tmp_path / "checkpoints" / name
        d.mkdir(parents=True)
        torch.manual_seed(seed)
        save_checkpoint(d / "best.pt", build_model(cfg), cfg, epoch=5)
    samples = tmp_path / "data" / "samples"
    samples.mkdir(parents=True)
    _scene(samples / "scene.tif")
    (tmp_path / "outputs").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(workspace, timeout: int = 120):
    """Execute app.py with ROOT pointed at the temporary workspace."""
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.session_state["_srm_root"] = str(workspace)
    return at.run()


def test_the_app_script_runs_without_raising(workspace, monkeypatch):
    """The regression. This failed with `TypeError: string indices must be integers`."""
    monkeypatch.setenv("SRM_ROOT", str(workspace))
    at = _run(workspace)
    assert not at.exception, [str(e) for e in at.exception]


def test_the_checkpoint_picker_is_populated(workspace, monkeypatch):
    monkeypatch.setenv("SRM_ROOT", str(workspace))
    at = _run(workspace)
    assert not at.exception, [str(e) for e in at.exception]
    labels = [w.label for w in at.sidebar.selectbox]
    assert any("Active checkpoint" in str(label) for label in labels)


def test_the_app_refuses_ai_enhancement_with_no_checkpoint(tmp_path, monkeypatch):
    """With nothing trained, it must say so -- and never fall back to bicubic."""
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SRM_ROOT", str(tmp_path))

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [str(e) for e in at.exception]
    text = " ".join(str(e.value) for e in at.error) + " ".join(str(m.value) for m in at.markdown)
    assert "No trained checkpoint" in text or "no trained checkpoint" in text.lower()
