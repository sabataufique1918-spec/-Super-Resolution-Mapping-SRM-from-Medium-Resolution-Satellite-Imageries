"""Super-Resolution Mapping (SRM) for medium-resolution satellite imagery.

Turns 10 m Sentinel-2 reflectance into <4 m products with deep generative
models, while keeping the result geospatially and spectrally faithful to the
observation it came from.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Sentinel-2 bands this project super-resolves. All four are natively 10 m,
# which is what makes a single-scale x4 model well posed.
S2_10M_BANDS = ("B02", "B03", "B04", "B08")
S2_BAND_WAVELENGTH_NM = {"B02": 492.4, "B03": 559.8, "B04": 664.6, "B08": 832.8}

# L2A surface reflectance is distributed as uint16 scaled by 10000.
S2_REFLECTANCE_SCALE = 10000.0

# Modulation Transfer Function at Nyquist for the Sentinel-2 MSI 10 m bands.
# Used to build a physically motivated blur kernel in the degradation model.
S2_MTF_NYQUIST = {"B02": 0.304, "B03": 0.300, "B04": 0.276, "B08": 0.234}

__all__ = [
    "__version__",
    "S2_10M_BANDS",
    "S2_BAND_WAVELENGTH_NM",
    "S2_REFLECTANCE_SCALE",
    "S2_MTF_NYQUIST",
]
