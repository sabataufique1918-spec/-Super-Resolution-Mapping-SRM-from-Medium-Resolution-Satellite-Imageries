# Sample scenes

Two **held-out test-split** tiles from SEN2NEON, staged so the dashboard has
something to run on immediately. Neither was seen during training: the split
is by NEON site, and both `BONA` and `NIWO` are test sites.

They are deliberately a contrasting pair. The model does not beat bicubic
everywhere, and picking only the scene where it wins would misrepresent it.

| | tile | site | CRS | S2↔NEON gap | reference nodata |
|---|---|---|---|---|---|
| A | `2021_BONA_4__3_5` | BONA (Alaska) | EPSG:32606 | 6 days | 0% |
| B | `2019_NIWO_3__4_4` | NIWO (Colorado) | EPSG:32613 | 1 day | 0% |

Each has an input and a reference:

```
<tile>_s2_10m.tif                 12-band Sentinel-2 L2A, 256x256 @ 10 m
<tile>_neon_2.5m_reference.tif    12-band NEON, 1024x1024 @ 2.5 m
```

RGB+NIR are file bands **2, 3, 4, 8**. The files carry no band descriptions
at all, so the layout is inferred from the band count and the assumption is
stated in the interface. Reading the first four bands instead gives NDVI
−0.35 where the truth is +0.89.

## Measured against the reference

`checkpoints/sen2neon_x4_detail/best.pt` (17 epochs, early-stopped, EMA
weights), CPU, 3.6 s per scene:

| metric | **A · BONA** bicubic → model | **B · NIWO** bicubic → model |
|---|---|---|
| PSNR (dB) | 30.879 → **31.996** | 39.949 → **40.119** |
| SSIM | 0.9573 → **0.9702** | 0.9730 → **0.9734** |
| RMSE | 0.0286 → **0.0251** | 0.0101 → **0.0099** |
| MAE | 0.0146 → **0.0135** | 0.0062 → 0.0066 |
| SAM (°) | 1.919 → **1.853** | 1.847 → 2.150 |

Ensemble disagreement across the two independently trained seeds:
**0.00093** mean on A, **0.00153** on B (reflectance units).

The gain is much larger where bicubic does worst. Per-site results for the
full 349-patch test split are in `outputs/eval_test_detail/report.md`.

Reproduce:

```bash
srm app                    # upload the pair in the sidebar
# or
srm infer --checkpoint checkpoints/sen2neon_x4_detail/best.pt \
  --input data/samples/2021_BONA_4__3_5_s2_10m.tif \
  --output outputs/bona_sr.tif --save-baseline --device cpu
```

Upload the reference in the sidebar to reproduce the accuracy table; without
one, the evaluation panel reports that reference-based evaluation is
unavailable rather than inventing a figure.

Source: SEN2NEON (Donike et al. 2026), CC-BY-4.0.
