"""Super-Resolution Mapping -- operator console.

A view, and only a view. Every model construction, raster read, metric and
export lives in `srm.service`, so the dashboard and the CLI cannot disagree
about what a number means.

Two behaviours are load-bearing rather than cosmetic:

* **Inference runs when Run is pressed, and at no other time.** The result is
  held in `st.session_state`, so moving a slider or switching a tab re-renders
  from the stored arrays instead of re-running the model. Streamlit re-executes
  this file top to bottom on every interaction, and a dashboard that
  super-resolves a scene each time someone adjusts a stretch is unusable.
* **Nothing is drawn from data that does not exist.** No reference means the
  evaluation panel says reference-based evaluation is unavailable; one
  checkpoint means the uncertainty panel says ensemble disagreement is
  unavailable. Neither substitutes a stand-in.

    srm app        # or: streamlit run app/app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm import service  # noqa: E402
from srm.ensemble import EnsembleResult  # noqa: E402
from srm.metrics.quality import METRIC_INFO  # noqa: E402

# The checkout this app serves. Overridable so the view can be executed
# against a temporary tree in tests -- see tests/test_app.py, which exists
# because a runtime error here is invisible to both the linter and every
# test of `srm.service`.
ROOT = Path(os.environ.get("SRM_ROOT") or Path(__file__).resolve().parents[1])
STAGING = Path(tempfile.gettempdir()) / "srm_uploads"
OUTPUTS = ROOT / "outputs" / "app"

st.set_page_config(
    page_title="SRM | Sentinel-2 Super-Resolution",
    page_icon="\U0001f6f0",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Numbers, coordinates and CRS strings are read character by character,
         so they are set in a monospace face and never wrap mid-token. */
      .srm-mono, .srm-kv td:last-child { font-family: "JetBrains Mono", "Cascadia Code",
          "SF Mono", Consolas, monospace; font-variant-numeric: tabular-nums; }
      .srm-hero { border-left: 3px solid #4c9aff; padding: .1rem 0 .1rem 1rem;
                  margin-bottom: 1.2rem; }
      .srm-hero h1 { font-size: 1.55rem; font-weight: 650; margin: 0; letter-spacing: -.01em; }
      .srm-hero p  { color: #8b949e; margin: .25rem 0 0; font-size: .9rem; }
      .srm-kv { width: 100%; border-collapse: collapse; font-size: .88rem; }
      .srm-kv td { padding: .34rem .6rem; border-bottom: 1px solid #21262d; }
      .srm-kv td:first-child { color: #8b949e; width: 46%; }
      .srm-kv td:last-child  { color: #e6edf3; text-align: right; }
      .srm-chip { display: inline-block; padding: .15rem .55rem; border-radius: 999px;
                  font-size: .74rem; font-weight: 600; letter-spacing: .02em;
                  border: 1px solid; margin-right: .35rem; }
      .chip-ok   { color:#3fb950; border-color:#23823a; background:#0f2416; }
      .chip-warn { color:#d29922; border-color:#9e6a03; background:#241c0f; }
      .chip-bad  { color:#f85149; border-color:#a13b35; background:#2b1416; }
      .chip-info { color:#4c9aff; border-color:#1f6feb; background:#0d1b2e; }
      .srm-panel { background:#0f141b; border:1px solid #21262d; border-radius:8px;
                   padding:.9rem 1.1rem; }
      .srm-note { color:#8b949e; font-size:.82rem; line-height:1.5; }
      section[data-testid="stSidebar"] { border-right: 1px solid #21262d; }
      div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# small rendering helpers (presentation only)
# --------------------------------------------------------------------------- #

def chip(text: str, kind: str = "info") -> str:
    return f'<span class="srm-chip chip-{kind}">{text}</span>'


def kv_table(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="srm-kv">{body}</table>', unsafe_allow_html=True)


def fmt(value, places: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:,.{places}f}{suffix}"


@st.cache_data(show_spinner=False)
def cached_checkpoints(root: str, _token: float) -> list[dict]:
    """`_token` busts the cache when Refresh is pressed; the scan reads every file.

    `root` is part of the key rather than closed over, so the cache cannot
    serve one checkout's checkpoints to another.
    """
    return service.discover_checkpoints([Path(root) / "checkpoints"])


@st.cache_data(show_spinner="Inspecting raster")
def cached_report(path: str, mtime: float):
    return service.describe_input(path)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

st.sidebar.markdown("### Super-Resolution Mapping")
st.sidebar.caption("Sentinel-2 10 m → 2.5 m · RGB + NIR · ×4")

st.sidebar.markdown("#### 1 · Input")
upload = st.sidebar.file_uploader(
    "Sentinel-2 GeoTIFF", type=["tif", "tiff"],
    help="A georeferenced multi-band raster. Bands are selected by name below.",
)
sample_paths = sorted((ROOT / "data").rglob("*.tif")) + sorted((ROOT / "outputs").rglob("*.tif"))
scene_path: str | None = None
if upload is not None:
    scene_path = str(service.stage_upload(upload.getbuffer(), upload.name, STAGING))
elif sample_paths:
    picked = st.sidebar.selectbox(
        "or a scene on disk", ["—"] + [str(p.relative_to(ROOT)) for p in sample_paths],
    )
    scene_path = str(ROOT / picked) if picked != "—" else None

reference_upload = st.sidebar.file_uploader(
    "Reference GeoTIFF (optional)", type=["tif", "tiff"],
    help="A higher-resolution image of the same ground. Without one, "
         "reference-based accuracy cannot be measured.",
)
reference_path = (
    str(service.stage_upload(reference_upload.getbuffer(),
                             f"ref_{reference_upload.name}", STAGING))
    if reference_upload is not None else None
)

st.sidebar.markdown("#### 2 · Model")
if st.sidebar.button("Rescan checkpoints", width="stretch"):
    st.session_state["ckpt_token"] = st.session_state.get("ckpt_token", 0) + 1
all_checkpoints = cached_checkpoints(str(ROOT), st.session_state.get("ckpt_token", 0))
usable = service.trained_checkpoints(all_checkpoints)

if not usable:
    st.sidebar.error("No trained checkpoint found.")
    st.markdown('<div class="srm-hero"><h1>Super-Resolution Mapping</h1>'
                "<p>Sentinel-2 10 m to 2.5 m, four bands, ×4</p></div>",
                unsafe_allow_html=True)
    st.error("**No trained checkpoint is available, so AI enhancement is disabled.**")
    st.markdown(
        "Bicubic enlargement is a separate, clearly labelled baseline and is never "
        "substituted for a model result. To produce a checkpoint:\n\n"
        "```bash\n"
        "python scripts/download_sen2neon.py --limit 150\n"
        "python scripts/prepare_sen2neon.py  --out data/patches_sen2neon\n"
        "python -m srm.train --config configs/sen2neon_x4.yaml\n"
        "```"
    )
    broken = [c for c in all_checkpoints if c.get("error")]
    if broken:
        st.warning("Files found under `checkpoints/` that could not be read:")
        for c in broken:
            st.code(f"{c['path']}\n{c['error']}")
    st.stop()


def label_checkpoint(info: dict) -> str:
    metrics = info.get("metrics") or {}
    psnr = metrics.get("psnr")
    tail = f" · {psnr:.2f} dB" if isinstance(psnr, (int, float)) else ""
    return f"{info['name']} · {info['architecture']} · ep{info['epoch']}{tail}"


# Widgets key their state by option value, so the options are the checkpoint
# *paths* -- hashable and stable across reruns -- and the record is looked up
# from them. Passing the dicts themselves makes selection state fragile.
by_path = {c["path"]: c for c in usable}

checkpoint = st.sidebar.selectbox(
    "Active checkpoint", list(by_path),          # keys are already the paths
    format_func=lambda p: label_checkpoint(by_path[p]),
    help="Only checkpoints recording at least one completed epoch are listed.",
)
checkpoint = by_path[checkpoint]

ensemble_extra = [
    by_path[p] for p in st.sidebar.multiselect(
        "Ensemble members (optional)",
        [c["path"] for c in usable if c["path"] != checkpoint["path"]],
        format_func=lambda p: label_checkpoint(by_path[p]),
        help="Independently trained checkpoints. Two or more in total enable "
             "the ensemble-disagreement map. Each member is a full pass over "
             "the scene, so the run takes proportionally longer.",
    )
]

st.sidebar.markdown("#### 3 · Bands")
report = cached_report(scene_path, Path(scene_path).stat().st_mtime) if scene_path else None
band_indices: list[int] = []
if report is not None and report.band_count:
    plan = service.plan_bands(report, checkpoint)
    if plan["error"]:
        st.sidebar.error(plan["error"])
    else:
        expected = plan["expected"]
        trained = checkpoint.get("bands") or [f"in{i+1}" for i in range(expected)]
        options = list(range(1, report.band_count + 1))
        st.sidebar.caption(
            "Map each model input to a band of the file. "
            "Defaults matched by name where possible."
        )
        for slot, default in zip(trained, plan["indices"]):
            band_indices.append(
                st.sidebar.selectbox(
                    f"{slot}", options, index=options.index(default),
                    format_func=lambda i: f"{i} — {report.band_names[i - 1]}",
                    key=f"band_{slot}",
                )
            )
        if len(set(band_indices)) != len(band_indices):
            st.sidebar.warning("The same band is mapped to more than one input.")
        for note in plan["notes"]:
            st.sidebar.caption(note)

st.sidebar.markdown("#### 4 · Processing")
import torch  # noqa: E402  -- deferred so the sidebar paints before torch loads

device = st.sidebar.selectbox(
    "Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
    help="Tiles are processed on this device; results are identical either way.",
)
tile = st.sidebar.select_slider("Tile size (input px)", [64, 96, 128, 192, 256, 384], value=128)
overlap = st.sidebar.select_slider("Tile overlap (input px)", [0, 8, 16, 24, 32, 48], value=16)
tta = st.sidebar.checkbox(
    "×8 self-ensemble", value=False,
    help="Average the eight flips and rotations. Eight times slower, usually "
         "0.1-0.3 dB better.",
)
save_outputs = st.sidebar.checkbox("Write products to outputs/app", value=True)

run_clicked = st.sidebar.button(
    "Run super-resolution", type="primary", width="stretch",
    disabled=not (scene_path and band_indices),
)


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #

st.markdown(
    '<div class="srm-hero"><h1>Super-Resolution Mapping</h1>'
    "<p>Deep-learning enhancement of medium-resolution satellite imagery "
    "· Sentinel-2 10 m → 2.5 m · blue, green, red, near-infrared</p></div>",
    unsafe_allow_html=True,
)

active = (
    chip(f"checkpoint {checkpoint['name']}", "info")
    + chip(f"{checkpoint['architecture']} ×{checkpoint['scale']}", "info")
    + chip(f"epoch {checkpoint['epoch']}", "info")
    + chip(f"seed {checkpoint.get('seed')}", "info")
    + chip(f"{checkpoint.get('weights', 'raw')} weights", "info")
    + (chip(f"ensemble {1 + len(ensemble_extra)} members", "ok")
       if ensemble_extra else chip("single model", "warn"))
)
st.markdown(active, unsafe_allow_html=True)

if scene_path is None:
    st.info(
        "Upload a Sentinel-2 GeoTIFF in the sidebar, or pick a scene from disk, "
        "to begin. The input is validated before anything is processed."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# 1 - input summary
# --------------------------------------------------------------------------- #

st.markdown("### 1 · Input")
left, right = st.columns([1, 1], gap="large")

with left:
    extent = report.ground_extent_m()
    kv_table([
        ("File", f"<span class='srm-mono'>{Path(report.path).name}</span>"),
        ("Dimensions", f"{report.width:,} × {report.height:,} px "
                       f"({report.megapixels:.2f} Mpx)"),
        ("Bands", f"{report.band_count} · {report.dtype}"),
        ("CRS", report.crs or "—"),
        ("Pixel spacing", fmt(report.pixel_size, 3, " m")),
        ("Ground extent", f"{fmt(extent[0], 0)} × {fmt(extent[1], 0)} m"
                          if extent else "—"),
        ("Valid data", f"{report.valid_fraction:.2%}"),
        ("Value range", f"{fmt(report.value_min, 4)} – {fmt(report.value_max, 4)}"),
    ])

with right:
    scale = int(checkpoint["scale"])
    out_h, out_w = report.output_shape(scale)
    kv_table([
        ("Output grid", f"{out_w:,} × {out_h:,} px"),
        ("Output pixel spacing", fmt(report.output_gsd(scale), 3, " m")),
        ("Scale factor", f"×{scale}"),
        ("Ground extent", "unchanged — the transform is divided, not resampled"),
    ])
    if report.ok:
        st.markdown(chip("input accepted", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(chip("input rejected", "bad"), unsafe_allow_html=True)

for message in report.errors:
    st.error(message)
for message in report.warnings:
    st.warning(message)

if not report.ok:
    st.stop()

st.markdown(
    '<p class="srm-note">A 2.5 m output grid is not a guarantee of 2.5 m '
    "resolved detail. The model estimates sub-pixel structure; only comparison "
    "against a true higher-resolution reference establishes how much of that "
    "estimate is real.</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# run - only on the button, never on a rerun
# --------------------------------------------------------------------------- #

if run_clicked:
    status = st.status("Processing", expanded=True)
    try:
        result = service.run_super_resolution(
            scene_path, checkpoint, band_indices, device=device, tile=tile,
            overlap=overlap, tta=tta,
            ensemble_paths=([checkpoint["path"]] + [c["path"] for c in ensemble_extra]
                            if ensemble_extra else None),
            progress=lambda m: status.write(m),
        )
        if reference_path:
            status.write("scoring against the reference")
            result.evaluation = service.evaluate_against_reference(result, reference_path)
        if save_outputs:
            status.write("writing products")
            service.write_outputs(result, OUTPUTS)
        status.update(label=f"Done in {result.seconds_model:.2f} s", state="complete",
                      expanded=False)
        st.session_state["result"] = result
    except Exception as exc:  # surfaced to the user, not swallowed
        status.update(label="Failed", state="error", expanded=True)
        st.error(f"**{type(exc).__name__}**\n\n{exc}")
        st.stop()

result = st.session_state.get("result")
if result is None:
    st.info("Press **Run super-resolution** in the sidebar to process this scene.")
    st.stop()

if result.report.path != report.path:
    st.warning(
        f"Showing results for `{Path(result.report.path).name}`, which is not the "
        "scene currently selected. Press Run to process the new one."
    )


# --------------------------------------------------------------------------- #
# 2 - comparison
# --------------------------------------------------------------------------- #

st.markdown("### 2 · Comparison")

bounds = service.shared_stretch(result.bicubic)
lr_rgb = service.to_rgb(result.lr, bounds)
bicubic_rgb = service.to_rgb(result.bicubic, bounds)
sr_rgb = service.to_rgb(result.sr, bounds)

st.caption(
    f"All three panels use one percentile stretch ({bounds[0]:.4f}–{bounds[1]:.4f} "
    "reflectance) so the comparison is fair. Display stretching is never applied "
    "to the exported data."
)

view = st.radio("View", ["Side by side", "Before / after slider"],
                horizontal=True, label_visibility="collapsed")

if view == "Side by side":
    a, b, c = st.columns(3)
    a.image(lr_rgb, caption=f"Input · {result.lr.shape[-1]}×{result.lr.shape[-2]} px "
                            f"· {fmt(result.report.pixel_size, 1)} m", width="stretch")
    b.image(bicubic_rgb, caption=f"Bicubic baseline · {result.sr.shape[-1]}"
                                 f"×{result.sr.shape[-2]} px "
                                 f"· {fmt(result.output_gsd, 2)} m", width="stretch")
    c.image(sr_rgb, caption=f"AI output · {result.sr.shape[-1]}"
                            f"×{result.sr.shape[-2]} px "
                            f"· {fmt(result.output_gsd, 2)} m", width="stretch")
else:
    try:
        from streamlit_image_comparison import image_comparison

        image_comparison(img1=bicubic_rgb, img2=sr_rgb,
                         label1="Bicubic baseline", label2="AI output",
                         width=1100, in_memory=True)
    except ImportError:
        a, b = st.columns(2)
        a.image(bicubic_rgb, caption="Bicubic baseline", width="stretch")
        b.image(sr_rgb, caption="AI output", width="stretch")
        st.caption("Install `streamlit-image-comparison` for the slider view.")

cols = st.columns(4)
cols[0].metric("Model time", f"{result.seconds_model:.2f} s",
               help=f"on {result.device}; bicubic control {result.seconds_bicubic:.2f} s")
cols[1].metric("Sharpness gain", f"{result.sharpness_gain:.3f}×",
               help="High-frequency energy relative to bicubic. Above 1 means detail "
                    "was added. It is not evidence the detail is correct.")
cols[2].metric("Output pixels", f"{result.sr.shape[-1] * result.sr.shape[-2] / 1e6:.2f} Mpx")
cols[3].metric("Wald consistency",
               fmt(result.consistency.get("consistency_rmse"), 5),
               help="Re-degrading the output should reproduce the input. Reference-free.")

st.markdown(
    '<p class="srm-note"><b>The bicubic panel is the control.</b> Any claim about '
    "the model is the difference between the two right-hand panels, not the "
    "appearance of the rightmost one. Bicubic is computed independently and is "
    "never substituted for a model result.</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# 3 - evaluation
# --------------------------------------------------------------------------- #

st.markdown("### 3 · Accuracy against a reference")
evaluation = result.evaluation

if evaluation and evaluation.get("available"):
    model_row = evaluation["metrics"]["model"]
    bicubic_row = evaluation["metrics"]["bicubic"]
    st.caption(
        f"Scored on {evaluation['valid_pixels']:,} pixels valid in the input, the "
        f"output and the reference — {evaluation['valid_fraction']:.1%} of the "
        f"output extent. {evaluation['note']}"
    )
    header = st.columns([1.4, 1, 1, 1, 1.6])
    for col, text in zip(header, ["Metric", "Bicubic", "AI model", "Better", "Unit"]):
        col.markdown(f"**{text}**")
    for key in service.REPORT_METRICS:
        if key not in model_row:
            continue
        info = METRIC_INFO.get(key, {"unit": "", "better": "lower", "note": ""})
        m, b = model_row[key], bicubic_row.get(key)
        row = st.columns([1.4, 1, 1, 1, 1.6])
        row[0].markdown(f"`{key}`")
        row[1].markdown(f"<span class='srm-mono'>{fmt(b, 4)}</span>", unsafe_allow_html=True)
        row[2].markdown(f"<span class='srm-mono'>{fmt(m, 4)}</span>", unsafe_allow_html=True)
        if b is None or not np.isfinite(b):
            row[3].markdown("—")
        else:
            wins = (m > b) if info["better"] == "higher" else (m < b)
            row[3].markdown(chip("model" if wins else "bicubic",
                                 "ok" if wins else "warn"), unsafe_allow_html=True)
        row[4].caption(f"{info['unit']} · {info['better']} is better")
    st.caption(
        f"PSNR and SSIM use an explicit data range of {evaluation['data_range']} "
        "(surface reflectance 0–1). SAM is in degrees. MAE and RMSE are in "
        "reflectance units."
    )
else:
    reason = (evaluation or {}).get(
        "reason", "No reference image was supplied.")
    st.markdown(
        f'<div class="srm-panel">{chip("reference-based evaluation unavailable", "warn")}'
        f'<p class="srm-note" style="margin-top:.6rem">{reason}<br><br>'
        "MAE, RMSE, PSNR, SSIM and spectral angle all measure a prediction "
        "<i>against a known truth</i>. Without a higher-resolution reference "
        "covering this ground there is nothing to measure against, so no accuracy "
        "figure is shown. The reference-free indicators above describe the output "
        "but do not establish that it is correct.</p></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 4 - uncertainty
# --------------------------------------------------------------------------- #

st.markdown("### 4 · Ensemble disagreement")

if result.ensemble is not None:
    ens = result.ensemble
    summary = ens.summary()
    st.caption(EnsembleResult.AGGREGATE_DEFINITION)

    vmax = float(np.nanpercentile(ens.disagreement, 99)) if np.isfinite(
        ens.disagreement).any() else 1.0
    a, b = st.columns([2, 1], gap="large")
    a.image(service.colorize(ens.disagreement, vmax=vmax),
            caption=f"Ensemble disagreement · black = members agree · "
                    f"bright = up to {vmax:.4f} reflectance apart", width="stretch")
    with b:
        kv_table([
            ("Members", str(summary["n_members"])),
            ("Mean", fmt(summary["disagreement_mean"], 5)),
            ("Median", fmt(summary["disagreement_median"], 5)),
            ("95th percentile", fmt(summary["disagreement_p95"], 5)),
            ("Maximum", fmt(summary["disagreement_max"], 5)),
            ("Units", "reflectance (0–1)"),
        ])
        st.caption("Per-band standard deviation, mean over the scene:")
        kv_table([(name, fmt(value, 5)) for name, value
                  in zip(result.product_band_names, summary["per_band_std_mean"])])

    with st.expander("Members"):
        for member in ens.members:
            st.markdown(
                f"- `{member['name']}` · {member['architecture']} · "
                f"epoch {member['epoch']} · seed {member['seed']}"
            )
    st.warning(
        "**This is disagreement between models, not a probability that the "
        "prediction is correct.** Members that share an architecture and a "
        "training set share their blind spots, so they can agree with each other "
        "and still all be wrong. Low disagreement is necessary for trust, not "
        "sufficient for it."
    )
else:
    reason = result.ensemble_error or (
        "Only one checkpoint was selected. Add independently trained checkpoints "
        "under **Ensemble members** in the sidebar."
    )
    st.markdown(
        f'<div class="srm-panel">{chip("ensemble uncertainty unavailable", "warn")}'
        f'<p class="srm-note" style="margin-top:.6rem">{reason}<br><br>'
        "Disagreement is measured <i>between independently trained models</i>. "
        "With a single model there is nothing to compare, and no substitute "
        "figure is shown here.</p></div>",
        unsafe_allow_html=True,
    )
    st.code(
        "python -m srm.train --config configs/sen2neon_x4.yaml \\\n"
        "  --set seed=2024 experiment=sen2neon_x4_s2024",
        language="bash",
    )


# --------------------------------------------------------------------------- #
# 5 - downloads
# --------------------------------------------------------------------------- #

st.markdown("### 5 · Downloads")
stem = Path(result.report.path).stem
written = result.written

cols = st.columns(4)

with cols[0]:
    path = written.get("enhanced_geotiff")
    if path and Path(path).is_file():
        st.download_button(
            "Enhanced GeoTIFF", Path(path).read_bytes(),
            file_name=Path(path).name, mime="image/tiff", width="stretch",
        )
        st.caption(f"4-band float32 reflectance, band order "
                   f"{', '.join(result.product_band_names)}, "
                   f"{result.report.crs}, {fmt(result.output_gsd, 2)} m.")
    else:
        st.caption("Enable **Write products** and re-run to export the GeoTIFF.")

with cols[1]:
    st.download_button(
        "RGB preview (PNG)", service.png_bytes(sr_rgb),
        file_name=f"{stem}_sr_preview.png", mime="image/png", width="stretch",
    )
    st.caption("Percentile-stretched for viewing only. Not scientific data.")

with cols[2]:
    path = written.get("uncertainty_raster")
    if path and Path(path).is_file():
        st.download_button(
            "Disagreement raster", Path(path).read_bytes(),
            file_name=Path(path).name, mime="image/tiff", width="stretch",
        )
        st.caption("Single-band float32, reflectance units, same grid as the output.")
    else:
        st.button("Disagreement raster", disabled=True, width="stretch")
        st.caption("Available when two or more checkpoints are ensembled.")

with cols[3]:
    path = written.get("report_markdown")
    data = (Path(path).read_text(encoding="utf-8") if path and Path(path).is_file()
            else service.format_run_report(result))
    st.download_button(
        "Evaluation report", data, file_name=f"{stem}_report.md",
        mime="text/markdown", width="stretch",
    )
    st.caption("Input, model, timing, accuracy and the stated limitations.")

if written.get("bicubic_geotiff"):
    st.caption(f"Bicubic baseline also written to "
               f"`{Path(written['bicubic_geotiff']).relative_to(ROOT)}`.")

with st.expander("Run record"):
    st.json(result.run_summary())
