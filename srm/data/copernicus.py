"""Copernicus Data Space Ecosystem client for Sentinel-2 L2A.

Two ways in, and this module supports both because they answer different needs:

* **Process API** (`sh.dataspace.copernicus.eu`) -- ask for a bounding box,
  a date range and an evalscript, get back a clipped, mosaicked, reprojected
  GeoTIFF of just the four 10 m bands. This is what you want for building a
  training set over many sites: a few MB per request instead of a 1 GB SAFE.

* **OData catalogue** (`catalogue.dataspace.copernicus.eu`) -- search and
  download whole products, for when you need the original SAFE tree, the
  scene classification layer, or full-tile processing.

Credentials come from the environment, never from code:

    CDSE_CLIENT_ID / CDSE_CLIENT_SECRET   (OAuth client, for the Process API)
    CDSE_USERNAME  / CDSE_PASSWORD        (account login, for OData download)

Register an OAuth client at https://shapps.dataspace.copernicus.eu/dashboard/
and browse the archive interactively at https://browser.dataspace.copernicus.eu
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"

# Returns the four native-10 m bands as float32 reflectance, with the scene
# classification layer available for cloud masking by the caller.
EVALSCRIPT_10M = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "SCL", "dataMask"], units: "REFLECTANCE" }],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.SCL, s.dataMask];
}
"""

# SCL classes that must not enter a training set: clouds, shadow, cirrus, snow.
SCL_REJECT = (3, 8, 9, 10, 11)
SCL_NODATA = (0,)


class CopernicusError(RuntimeError):
    pass


@dataclass
class AOI:
    """A named request: bounding box in WGS84 plus a date window."""

    name: str
    bbox: tuple[float, float, float, float]   # min_lon, min_lat, max_lon, max_lat
    start: str                                 # "2024-05-01"
    end: str                                   # "2024-09-30"
    max_cloud: float = 10.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "start": self.start,
            "end": self.end,
            "max_cloud": self.max_cloud,
        }


def _requests():
    try:
        import requests

        return requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install requests") from exc


def get_token(client_id: str | None = None, client_secret: str | None = None) -> str:
    """Client-credentials OAuth token for the Process API."""
    requests = _requests()
    client_id = client_id or os.environ.get("CDSE_CLIENT_ID")
    client_secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise CopernicusError(
            "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET. Create an OAuth client at "
            "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings"
        )
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise CopernicusError(f"token request failed [{response.status_code}]: {response.text[:300]}")
    return response.json()["access_token"]


def _bbox_size_pixels(bbox: Sequence[float], resolution_m: float) -> tuple[int, int]:
    """Approximate pixel dimensions of a WGS84 bbox at a given ground resolution."""
    import math

    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2.0)
    metres_per_deg_lat = 111_132.0
    metres_per_deg_lon = 111_320.0 * math.cos(mid_lat)
    width = int(round((max_lon - min_lon) * metres_per_deg_lon / resolution_m))
    height = int(round((max_lat - min_lat) * metres_per_deg_lat / resolution_m))
    # The Process API caps a single response at 2500 px per side.
    return max(1, min(width, 2500)), max(1, min(height, 2500))


def fetch_s2_patch(
    aoi: AOI,
    out_path: str | Path,
    resolution_m: float = 10.0,
    token: str | None = None,
    mosaicking: str = "leastCC",
    retries: int = 3,
) -> Path:
    """Download one AOI as a 6-band GeoTIFF (B02, B03, B04, B08, SCL, dataMask).

    `mosaicking="leastCC"` asks the server for the least-cloudy pixel in the
    window, which is usually what you want for a clean training scene.
    """
    requests = _requests()
    token = token or get_token()
    width, height = _bbox_size_pixels(aoi.bbox, resolution_m)
    payload = {
        "input": {
            "bounds": {
                "bbox": list(aoi.bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{aoi.start}T00:00:00Z",
                            "to": f"{aoi.end}T23:59:59Z",
                        },
                        "maxCloudCoverage": aoi.max_cloud,
                        "mosaickingOrder": mosaicking,
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT_10M,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = ""
    for attempt in range(retries):
        response = requests.post(
            PROCESS_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"},
            timeout=300,
        )
        if response.status_code == 200:
            out_path.write_bytes(response.content)
            return out_path
        last_error = f"[{response.status_code}] {response.text[:300]}"
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt * 5)  # server-side throttling: back off
            continue
        break
    raise CopernicusError(f"process request for {aoi.name} failed: {last_error}")


def search_products(
    bbox: Sequence[float],
    start: str,
    end: str,
    max_cloud: float = 10.0,
    product_type: str = "S2MSI2A",
    top: int = 50,
) -> list[dict[str, Any]]:
    """OData catalogue search. Needs no credentials -- useful for planning."""
    requests = _requests()
    min_lon, min_lat, max_lon, max_lat = bbox
    polygon = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
        f"{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    filters = [
        "Collection/Name eq 'SENTINEL-2'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
        f"ContentDate/Start gt {start}T00:00:00.000Z",
        f"ContentDate/Start lt {end}T23:59:59.999Z",
        (
            "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
            f"and a/Value eq '{product_type}')"
        ),
        (
            "Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' "
            f"and a/Value le {max_cloud})"
        ),
    ]
    response = requests.get(
        CATALOG_URL,
        params={"$filter": " and ".join(filters), "$top": top, "$orderby": "ContentDate/Start asc"},
        timeout=120,
    )
    if response.status_code != 200:
        raise CopernicusError(f"catalogue search failed [{response.status_code}]: {response.text[:300]}")
    return response.json().get("value", [])


def scl_valid_mask(scl: "Any", reject: Sequence[int] = SCL_REJECT) -> "Any":
    """Boolean mask of pixels safe to train on, from the scene classification layer."""
    import numpy as np

    scl = np.asarray(scl)
    invalid = np.isin(np.rint(scl).astype(np.int16), list(reject) + list(SCL_NODATA))
    return ~invalid


def load_s2_geotiff(path: str | Path, min_valid_fraction: float = 0.98):
    """Read a `fetch_s2_patch` product into (reflectance (4, H, W), valid mask).

    Raises if too much of the window is cloudy: a training scene with cloud in
    it teaches the model to hallucinate cloud texture into clear imagery.
    """
    import numpy as np

    from ..utils.geo import read_raster

    array, meta = read_raster(path, as_reflectance=False)
    if array.shape[0] < 6:
        raise CopernicusError(
            f"{path} has {array.shape[0]} bands; expected 6 (B02,B03,B04,B08,SCL,dataMask)"
        )
    reflectance = array[:4]
    valid = scl_valid_mask(array[4]) & (array[5] > 0.5) & np.isfinite(reflectance).all(axis=0)
    fraction = float(valid.mean())
    if fraction < min_valid_fraction:
        raise CopernicusError(
            f"{Path(path).name}: only {fraction:.1%} of pixels are clear "
            f"(need {min_valid_fraction:.0%}); pick another date or AOI"
        )
    meta.band_names = ["B02", "B03", "B04", "B08"]
    return np.clip(reflectance, 0.0, 1.5).astype(np.float32), valid, meta


# A spread of land-cover regimes, so a model is not tuned to one landscape.
DEFAULT_AOIS: list[AOI] = [
    AOI("nl_flevoland_agri", (5.35, 52.45, 5.60, 52.60), "2024-06-01", "2024-08-31"),
    AOI("in_pune_urban", (73.78, 18.46, 73.98, 18.60), "2024-01-01", "2024-03-31"),
    AOI("es_ebro_delta", (0.62, 40.66, 0.85, 40.78), "2024-05-01", "2024-07-31"),
    AOI("in_punjab_fields", (75.72, 30.82, 75.95, 30.96), "2024-10-01", "2024-12-15"),
    AOI("de_leipzig_mixed", (12.26, 51.28, 12.48, 51.40), "2024-04-01", "2024-06-30"),
    AOI("eg_nile_valley", (31.10, 29.95, 31.32, 30.10), "2024-02-01", "2024-04-30"),
]
