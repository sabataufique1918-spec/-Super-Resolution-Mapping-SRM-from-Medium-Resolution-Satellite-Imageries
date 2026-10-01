# Super-Resolution Mapping (SRM) from Medium-Resolution Satellite Imagery

Turns Sentinel-2 10 m surface reflectance into a 2.5 m product with deep
super-resolution models, while keeping the result **geospatially and
spectrally faithful to the observation it came from**.

The distinction matters. Most super-resolution work optimises for how an image
*looks*. A remote-sensing product is measured, not viewed: it feeds a
classifier, an index time series, or a change detector. So this repository
treats spectral integrity and observation-consistency as first-class
objectives, reports an interpolation baseline on every metric, and ships
per-pixel uncertainty maps that mark where the model invented detail the
sensor never recorded.

---

## What it does

| Stage | Module | Output |
|---|---|---|
| Acquire | `srm/data/sen2neon.py` | **real** 10 m Sentinel-2 / 2.5 m NEON pairs |
| Acquire | `srm/data/copernicus.py` | cloud-screened Sentinel-2 L2A GeoTIFFs |
| Degrade | `srm/data/degradation.py` | simulated LR/HR pairs (MTF-based PSF) |
| Validate | `srm/validation.py` | input admission control + the CRS/extent contract |
| Train | `srm/train.py` | masked L1 + Adam, EMA, early stopping, resume |
| Evaluate | `srm/evaluate.py` | MAE, RMSE, PSNR, SSIM, SAM, time — overall and by area |
| Infer | `srm/infer.py` | tiled, seam-free, georeferenced 2.5 m GeoTIFF |
| Qualify | `srm/ensemble.py` | ensemble disagreement across independent checkpoints |
| Qualify | `srm/uncertainty.py` | single-model reference-free indicators |
| Serve | `srm/service.py` | the layer the dashboard and CLI share |
| Apply | `srm/applications/` | spectral indices, land cover, change detection |

Bands are the four native-10 m Sentinel-2 channels — **B02, B03, B04, B08** —
which is what makes a single-scale x4 model well posed. No pan-sharpening
assumptions, no mixed-GSD resampling.

> Bands are selected **by name**, never by position. SEN2NEON ships 12 bands
> (`B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12`, with B10 absent), so RGB+NIR are
> file bands **4, 3, 2, 8**. Taking the first four gives coastal-aerosol,
> blue, green and red instead — on a forest tile that turns a correct NDVI of
> **+0.90** into **−0.37**, with no error anywhere. `band_indices()` exists to
> make that mistake impossible.

### The reference architecture

`model.name: residual_cnn` is a configurable residual CNN, and the default:

```
[B,4,64,64] -> 3x3 conv 4->64
            -> 8 x { 3x3 conv, ReLU, 3x3 conv, + local residual }
            -> 3x3 conv, + global feature skip from the head
            -> 2 x { 3x3 conv 64->256, PixelShuffle(2), ReLU }
            -> 3x3 conv 64->4                       -> [B,4,256,256]
```

Unlike every other generator here it has **no bicubic image-domain skip** —
the tail convolution emits the image. That is a real difference, not a detail:
the model must learn the low frequencies as well as the detail, it starts
further from a usable output, and it needs its own initialisation. `blocks`,
`features` and `res_scale` are all config keys.

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install -e .

srm demo
```

> **If this checkout lives inside OneDrive, Dropbox or similar, put the
> environment somewhere else** — `python -m venv ~/.venvs/srm`. A few GB of
> CUDA DLLs in a synced folder are uploaded continuously and can be locked
> mid-write; `.gitignore` does not stop that. The same goes for the dataset:
> pass `--root` outside the synced tree.

---

## Moving this to another machine

Nothing in the code hardcodes a path — dataset locations default to
`~/srm-data/sen2neon` and everything else is relative to the checkout. What
does *not* travel is the virtual environment (its paths are absolute) and
the SEN2NEON download (it lives outside the project on purpose).

**What to copy.** The folder is ~970 MB in full, but most of that is the
training set, which is only needed to train:

| | size | needed for |
|---|---:|---|
| code, `configs/`, `tests/` | ~2 MB | everything |
| `checkpoints/` | 69 MB | running the model at all |
| `data/samples/` | 22 MB | the bundled demo scenes |
| `data/patches_sen2neon/` | 726 MB | **training only** — skip it |
| `outputs/` | 154 MB | nothing — regenerated |

So a ~95 MB bundle (everything except `data/patches_sen2neon/` and
`outputs/`) runs the dashboard, inference, evaluation and the ensemble.

**On the new machine:**

```bash
python -m venv ~/.venvs/srm          # outside any synced folder
~/.venvs/srm/Scripts/activate        # Windows; source .../bin/activate elsewhere
pip install -r requirements.txt
pip install -e .

srm check                            # verify before anything else
streamlit run app/app.py
```

`srm check` is the one to run first. It reports the interpreter, every
platform-specific wheel, whether a GPU was found, which checkpoints are
present and loadable, and then super-resolves a bundled scene and verifies
the CRS and ground extent survived. It exits non-zero if anything required
is missing, and tells you the command to fix it.

**A GPU is optional.** Without one everything falls back to CPU: about 4
seconds for a 256×256 tile, which is fine for the dashboard. Training is the
only part that really wants a GPU.

**Python 3.10+.** Verified here on 3.14.4 with torch 2.14; 3.11–3.13 have the
widest wheel coverage for rasterio and GDAL if the install fights back.

`srm demo` runs the entire pipeline on procedurally generated scenes — build,
train, evaluate, super-resolve, quantify uncertainty, and run all three
downstream applications — in about **50 seconds on a GPU**, a few minutes on
CPU. It needs no credentials and no downloads.

It writes to `outputs/demo/`:

```
report.md                 consolidated findings
eval/report.md            full accuracy assessment vs the bicubic baseline
figures/*.png             LR / bicubic / SR / reference comparisons
test_scene_sr.tif         a georeferenced 2.5 m product
uncertainty/*.tif         per-pixel reliability
```

> **The demo's numbers measure the pipeline, not Sentinel-2.** The input is
> synthetic, so the code path is real but the imagery is not. To make a claim
> about Sentinel-2, run the real-data path below.

---

## Commands

Every step is available through one entry point. `srm <command> --help` gives
that command's own options.

```bash
# real data: Sentinel-2 10 m / NEON 2.5 m pairs
srm fetch-sen2neon --limit 150
srm prepare-sen2neon --out data/patches_sen2neon
srm overfit  --config configs/sen2neon_x4.yaml
srm train    --config configs/sen2neon_x4.yaml
srm eval     --checkpoint checkpoints/sen2neon_x4/best.pt \
             --patches data/patches_sen2neon --split test
srm infer    --checkpoint checkpoints/sen2neon_x4/best.pt --input scene.tif --output sr.tif
srm ensemble --checkpoints a/best.pt b/best.pt --input scene.tif --outdir unc/

# simulated data, no downloads or credentials
srm demo
srm download --list-aois
srm prepare  --source geotiff --glob "data/raw/*.tif"
srm uncertainty --checkpoint ... --input scene.tif --outdir unc/

srm models                     # list architectures
srm app                        # Streamlit dashboard
```

Each is equivalently `python -m srm.train ...` and so on, so nothing is hidden
behind the CLI.

---

## The real-data path (recommended): SEN2NEON

This is the path whose numbers are reportable, because the low-resolution
side is an **actual Sentinel-2 observation** rather than a degradation of the
reference. [SEN2NEON](https://huggingface.co/datasets/isp-uv-es/SEN2NEON)
(CC-BY-4.0, Donike et al. 2026) pairs Sentinel-2 L2A chips with NEON
AVIRIS-NG imagery convolved to the Sentinel-2 spectral response functions:

| | product | shape | nodata |
|---|---|---|---|
| LR | `s2_l2a_10m/` | 12 × 256 × 256 @ 10 m | 65535 |
| HR | `neon_2.5m_linearized/` | 12 × 1024 × 1024 @ 2.5 m | 0 |

Both uint16 scaled by 10000, same UTM CRS per tile, pixel-aligned, exactly 4×
apart. Use `s2_l2a_10m/` — the 20 Aug 2026 release corrected an earlier upload
that shipped a linearized 10 m NEON product in its place.

```bash
srm fetch-sen2neon --metadata-only          # inspect the index first
srm fetch-sen2neon --limit 150              # ~4 GB, spread over ~46 NEON sites
srm prepare-sen2neon --out data/patches_sen2neon
srm overfit --config configs/sen2neon_x4.yaml     # verify the training path
srm train   --config configs/sen2neon_x4.yaml
srm eval    --checkpoint checkpoints/sen2neon_x4/best.pt \
            --patches data/patches_sen2neon --split test
```

Point `--root` outside any synced folder. Several GB of GeoTIFFs inside
OneDrive or Dropbox will be uploaded continuously and can be locked mid-write;
`.gitignore` does not stop that.

### How the data is split

**70 / 15 / 15 by NEON site**, never by patch. Tiles named `..._3__0_2` and
`..._3__1_2` are adjacent crops of one mosaic: they share a border, a flight
line and an atmosphere. Splitting per patch puts a validation crop next to its
training neighbour and inflates every metric. Splitting per *site* also
separates repeat observations of one location across years. The split report
(`data/patches_sen2neon/split_report.json`) carries the leakage check, and
`prepare-sen2neon` refuses to continue if any site appears in two splits.

### What limits these numbers

Recorded per tile and repeated in every generated report:

- **Temporal separation** — median 23 days between the NEON overflight and the
  Sentinel-2 acquisition, up to 40. Some residual is real surface change.
- **Cross-sensor radiometry** — the HR side is AVIRIS-NG convolved to
  Sentinel-2 SRFs, not a Sentinel-2 observation.
- **Cloud information is per tile, not per pixel.** The index carries
  `cloud_score_plus_cdf`; the 12-band stack has no SCL layer, so tiles are
  filtered on the scalar and **no per-pixel cloud mask is applied**.
- **Coverage is North American NEON sites**, skewed to natural land cover.

### Masking is not optional here

NEON flight lines leave large empty wedges — up to 50% of a tile, ~10% on
average. Those pixels are stored as zeros. An unmasked L1 loss reads them as a
consistent signal that scene corners are black, and the model learns exactly
that. Every patch therefore carries a validity mask, and the pixel loss
averages over observed pixels only (`srm/losses/masked.py`).

---

## The simulated path: Copernicus + Wald

### 1. Credentials

Register an OAuth client at the
[Copernicus Data Space](https://dataspace.copernicus.eu/), then set:

```bash
export CDSE_CLIENT_ID=...          # set CDSE_CLIENT_ID=... on Windows
export CDSE_CLIENT_SECRET=...
```

Credentials are read from the environment only, never from code or config.

### 2. Download

```bash
srm download --list-aois           # six built-in AOIs: agriculture, urban, delta
srm download --aoi nl_flevoland_agri in_pune_urban
srm download --bbox 73.78 18.46 73.98 18.60 --name my_aoi --start 2024-01-01 --end 2024-03-31
```

Scenes are cloud-screened using the L2A scene-classification layer and
rejected if too little of the AOI is usable.

### 3. Build pairs and train

```bash
srm prepare --source geotiff --glob "data/raw/*.tif" --out data/patches
srm train   --config configs/edsr_x4.yaml
srm eval    --checkpoint checkpoints/edsr_x4/best.pt --patches data/patches
```

**Where the high-resolution truth comes from.** There is no 2.5 m Sentinel-2.
Training pairs are made by *degrading* the 10 m observation to 40 m with a
sensor-motivated PSF and learning 40 m to 10 m, then applying that mapping at
10 m to 2.5 m. This is the standard Wald protocol, and it carries a standing
assumption: that the scaling behaviour learned at one ratio transfers to the
next. The `consistency_*` metrics are what test it — they degrade the
super-resolved output and check that it reproduces the observed input.

---

## Reading the results

An SR result without an interpolation baseline on the same split is not a
measurable claim, so `srm eval` always reports both:

| Metric | Measures | Watch for |
|---|---|---|
| PSNR / SSIM / RMSE | pixel fidelity | a model that merely blurs can win these |
| SAM (degrees) | spectral distortion | under ~2° preserves land-cover signature |
| ERGAS | standard resolution-fusion summary | under 3 conventionally reads as good |
| **SCC** | whether detail *correlates with truth* | the one that catches hallucination |
| **consistency_\*** | Wald: degrade(SR) must reproduce LR | the physical sanity check |
| sharpness_gain | high-frequency energy vs bicubic | >1 with flat SCC means invented texture |

SCC and consistency are the ones that distinguish a super-resolution product
from a plausible-looking picture.

Reported per geographic area as well as overall. A model can gain 0.4 dB on
average while losing 1 dB on every built-up tile, because natural land cover
dominates the set; only the per-area table shows that.

Every metric is reported with its unit and direction, and PSNR and SSIM with
an **explicit data range** (1.0, surface reflectance). A PSNR computed against
a per-patch maximum flatters dark scenes and is not comparable between
datasets.

### What a 2.5 m grid does and does not mean

The output is written on a 2.5 m pixel grid. That is not the same as 2.5 m of
resolved detail. The model produces a plausible estimate of sub-pixel
structure; how much of it is real can only be established by comparison
against a true higher-resolution reference. Where no reference exists, the
output is a hypothesis rendered with the visual authority of an observation —
which is precisely why the bicubic control and the disagreement map are not
optional parts of the product.

### Uncertainty: two different things, not interchangeable

**`srm ensemble`** — disagreement across *independently trained* checkpoints.
Per-band standard deviation across members, aggregated as their RMS, in
reflectance units. This is the one the specification asks for, and it is
labelled **ensemble disagreement**, never confidence or probability:

```bash
srm train --config configs/sen2neon_x4.yaml --set seed=2024 experiment=sen2neon_x4_s2024
srm ensemble --checkpoints checkpoints/sen2neon_x4/best.pt \
                           checkpoints/sen2neon_x4_s2024/best.pt \
             --input scene.tif --outdir outputs/unc
```

With one checkpoint it reports **"ensemble disagreement unavailable"** and
computes nothing. There is no single-model substitute, because models that
share an architecture and a training set share their blind spots: they can
agree with each other and still all be wrong. Low disagreement is necessary
for trust, not sufficient.

**`srm uncertainty`** — single-model, reference-free indicators: MC-dropout or
TTA spread, Wald observation inconsistency, and inferred-detail energy. These
describe one model's behaviour and answer a different question.

---

## The dashboard

```bash
srm app                        # or: streamlit run app/app.py
```

A dark scientific console with five panels: input summary (dimensions, CRS,
pixel spacing, valid coverage, output grid), a three-way comparison
(input / bicubic / AI) on **one shared stretch** with an optional before-after
slider, measured accuracy against an uploaded reference, ensemble
disagreement, and downloads.

Behaviours worth knowing:

- Inference runs **only when Run is pressed**. Results live in session state,
  so moving a slider re-renders from the stored arrays rather than
  re-running the model.
- All model and raster code lives in `srm/service.py`. The UI file contains
  no inference, no rasterio and no metrics, so the dashboard and the CLI
  cannot disagree about a number.
- With no trained checkpoint the app refuses to run AI enhancement and says
  so. **Bicubic is never silently substituted for a model result.**
- No reference means the evaluation panel explains that reference-based
  evaluation is unavailable. One checkpoint means the uncertainty panel says
  ensemble disagreement is unavailable. Neither invents a stand-in.
- Display stretching is confined to the screen and the PNG preview. Exported
  GeoTIFFs carry unstretched reflectance.

---

## Model choice

```
residual_cnn  the specified reference architecture -- default; alias srmnet
bicubic       zero-parameter control; must be reported
srcnn         weak learned baseline
edsr          PSNR-oriented workhorse with a bicubic skip
rrdb          higher capacity, generator for the adversarial stage
esrgan        same trunk as rrdb; the difference is the loss configuration
swinir        transformer, sized to fit a 4 GB GPU at 64x64 LR patches
```

**`residual_cnn`** is the specified network and the default. Its one
structural difference from the rest is that it has no bicubic image-domain
skip, so it must learn the low frequencies as well as the detail. That is a
slower start by construction: `global_skip: bicubic` switches on the skip for
a controlled comparison, and the initialisation follows automatically
(a damped tail for the skip variant, ordinary Kaiming without it).

Use a fidelity model when the output feeds a classifier, an index, or a change
detector: its errors are honest blur rather than invented structure. Reach for
the adversarial configuration only when a human will look at the image, and
read the disagreement maps when you do.

The configs implement a two-stage recipe. Train the fidelity model first, then
fine-tune with perceptual and adversarial terms via `train.init_from` —
starting a GAN from scratch on satellite imagery reliably produces colour
drift and texture with no basis in the data.

---

## Configuration

Everything is YAML; nothing about an experiment lives in code.

```bash
srm train --config configs/edsr_x4.yaml --set train.epochs=20 --set model.features=96
```

| Config | Purpose |
|---|---|
| `configs/sen2neon_x4.yaml` | **the reference configuration** — real pairs, masked L1, Adam |
| `configs/demo_fast.yaml` | smoke test and CI — a working model, not a good one |
| `configs/edsr_x4.yaml` | bicubic-skip fidelity model on simulated pairs |
| `configs/esrgan_x4.yaml` | perceptual fine-tune, initialised from the above |
| `configs/swinir_x4.yaml` | transformer variant |

Checkpoints embed their own config **and a provenance record** — band order,
reflectance scaling, normalisation, scale factor, seed and the dataset they
were trained on. Inference never has to be told which architecture to load,
and a checkpoint whose band order is unknown cannot be applied by accident.

Selection uses masked validation **loss** by default (`train.monitor`), with
`early_stopping_patience` and `resume` available.

### Three silent-failure guards

Each of these is a constant tuned for long runs that quietly produces a bad
result on a short one, with no error anywhere:

1. **`train.warmup_iters`** is clamped to 25% of the run's total steps. The
   bicubic-skip models initialise their residual near zero, so a run too
   short to clear its own warmup yields a checkpoint numerically
   indistinguishable from bicubic — which the report then presents as a
   result.
2. **`train.ema_decay`** is clamped so the EMA horizon is at most a tenth of
   the run, **and** the choice between raw and EMA weights is settled by
   measurement at the end of training, not assumed. At decay 0.999 the
   average spans ~1000 steps; on a run that is still improving when it ends
   that averages in strictly worse earlier weights. Measured here: the EMA
   scored **36.45 dB / SAM 3.27°** where the raw weights scored
   **37.80 dB / SAM 1.62°** — the difference between a model that beats
   bicubic and one that appears not to, reported nowhere. `load_checkpoint`
   now honours the `prefer` key that the run stamps in; use
   `scripts/stamp_weights.py` for checkpoints made before that existed.
3. **Output clamping is applied at inference only.** `clamp` has exactly zero
   gradient outside its bounds, so clamping during training lets a direct-
   output model that starts below zero detach from the optimiser entirely and
   sit at the floor forever. This was caught by `srm overfit`, not by review.

If you see guard 1 or 2 warn, the run is too short for its settings.

### Verify before you commit hours

```bash
srm overfit --config configs/sen2neon_x4.yaml
```

Memorising eight patches takes a minute and catches an inverted mask, an
off-by-a-crop LR/HR pairing, a dead output head or an unusable learning rate.
It compares against bicubic on the same patches and fails if the model cannot
beat it. Passing means the plumbing works — it says nothing about
generalisation.

---

## Development

```bash
pip install -e .
pytest
```

Tests cover the geometry and spectral invariants that are easy to break and
expensive to notice: degradation and upsample round-trips, GeoTIFF CRS and
transform preservation, tiled inference seam-freeness, metric correctness
against known-answer cases, and the LR-schedule guard above.

---

## Layout

```
srm/
  data/         sen2neon (real pairs), copernicus, degradation, patches, datasets
  models/       residual_cnn (specified) + others, registry, checkpoint I/O
  losses/       masked pixel losses, spectral-angle, Wald consistency, GAN
  metrics/      quality metrics + units, shared by training and evaluation
  applications/ indices, land cover, change detection
  utils/        geo, viz, logging, seeding
  validation.py input admission control and band mapping
  service.py    orchestration shared by the dashboard and the CLI
  ensemble.py   disagreement across independently trained checkpoints
  uncertainty.py single-model reference-free indicators
configs/        experiment definitions
scripts/        download_sen2neon, prepare_sen2neon, overfit_check, demo
app/            Streamlit dashboard (a view only -- no model code)
tests/          222 tests
```

## License

MIT.
