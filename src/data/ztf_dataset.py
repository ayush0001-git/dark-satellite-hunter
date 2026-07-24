"""
ZTF Dataset Loader and Preprocessing (`ztf_dataset.py`)

* `RobustScalerPerBand`  - median/IQR normalization per photometric band.
* `PolarsZTFDataset`     - out-of-core scanning (Polars) of real survey
                           archives: .parquet / .csv files or directories,
                           with automatic column-alias normalization.
* `LightCurveDataset`    - fixed-length windowing/padding into model tensors,
                           with an observation mask so padded epochs can be
                           excluded from the reconstruction loss.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl
from typing import Dict, List, Optional, Tuple, Union


class RobustScalerPerBand:
    """Median and IQR normalization applied independently per band."""

    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.stats: Dict = {}

    def fit_transform(
        self, mag: np.ndarray, magerr: np.ndarray, bands: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm_mag = np.zeros_like(mag, dtype=np.float32)
        norm_magerr = np.zeros_like(magerr, dtype=np.float32)

        for b in np.unique(bands):
            mask = bands == b
            b_mag = mag[mask]
            median = np.median(b_mag)
            q75, q25 = np.percentile(b_mag, [75, 25])
            iqr = q75 - q25 + self.eps
            self.stats[b] = {"median": float(median), "iqr": float(iqr)}
            norm_mag[mask] = (b_mag - median) / iqr
            norm_magerr[mask] = magerr[mask] / iqr

        return norm_mag, norm_magerr


class PolarsZTFDataset:
    """
    Out-of-core scanner for real survey light-curve archives.

    Accepts:
        * a single `.parquet` or `.csv` file,
        * or a directory containing any mix of the two.

    Column names are normalized through an alias map so common ZTF/IRSA/
    Kaggle exports work out of the box (`objid` -> object_id, `magpsf` -> mag,
    `fid`/`filtercode` -> filter, ...). Julian Dates are auto-converted to
    MJD. Only g/r epochs are kept (i-band and others are dropped). Yields
    per-object dictionaries in the same schema the mock generator produces,
    so the rest of the pipeline is agnostic to the data source.
    """

    REQUIRED_COLUMNS = ("object_id", "mjd", "mag", "magerr", "filter")
    COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
        "object_id": ("object_id", "objid", "objectid", "oid", "source_id", "id"),
        "mjd": ("mjd", "obsmjd", "hjd", "jd", "time", "mjd_obs"),
        "mag": ("mag", "magpsf", "psfmag", "magnitude"),
        "magerr": ("magerr", "sigmapsf", "magerr_psf", "mag_err", "e_mag", "err"),
        "filter": ("filter", "fid", "filtercode", "band", "passband"),
        "ra": ("ra", "radeg", "ra_deg"),
        "dec": ("dec", "decdeg", "dec_deg", "decl"),
    }

    def __init__(self, data_path: str, seq_len: int = 256, min_obs: int = 10):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Light-curve archive not found: {data_path}")
        self.data_path = data_path
        self.seq_len = seq_len
        self.min_obs = min_obs

        lf = self._scan()
        raw_cols = lf.collect_schema().names()
        self._rename_map = self._build_rename_map(raw_cols)

        resolved = set(self._rename_map.values())
        missing = [c for c in self.REQUIRED_COLUMNS if c not in resolved]
        if missing:
            raise ValueError(
                f"Archive is missing required columns {missing} "
                f"(found: {raw_cols}). Recognized aliases: "
                + "; ".join(f"{k}: {v}" for k, v in self.COLUMN_ALIASES.items())
            )

    def _scan(self) -> pl.LazyFrame:
        """Builds a lazy frame from a file or a directory of parquet/csv files."""
        path = self.data_path
        if os.path.isdir(path):
            frames = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if name.lower().endswith(".parquet"):
                    frames.append(pl.scan_parquet(full))
                elif name.lower().endswith(".csv"):
                    frames.append(pl.scan_csv(full, infer_schema_length=10000))
            if not frames:
                raise FileNotFoundError(f"No .parquet or .csv files inside {path!r}")
            return pl.concat(frames, how="diagonal_relaxed")
        if path.lower().endswith(".csv"):
            return pl.scan_csv(path, infer_schema_length=10000)
        return pl.scan_parquet(path)

    @classmethod
    def _build_rename_map(cls, raw_cols: List[str]) -> Dict[str, str]:
        """Maps raw column names to the canonical schema (first alias wins)."""
        lower = {c.lower(): c for c in raw_cols}
        rename: Dict[str, str] = {}
        for canonical, aliases in cls.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower and lower[alias] not in rename:
                    rename[lower[alias]] = canonical
                    break
        return rename

    def _load(self, object_ids: Optional[List[Union[int, str]]] = None) -> pl.DataFrame:
        lf = self._scan().rename(self._rename_map).select(list(set(self._rename_map.values())))
        if object_ids is not None:
            lf = lf.filter(pl.col("object_id").is_in(object_ids))
        df = lf.sort(["object_id", "mjd"]).collect()

        # Auto-convert Julian Date to Modified Julian Date if needed.
        if len(df) and float(df["mjd"].median()) > 2_400_000.0:
            df = df.with_columns((pl.col("mjd") - 2_400_000.5).alias("mjd"))
        return df

    def get_object_ids(self, limit: Optional[int] = None) -> List[Union[int, str]]:
        df = self._scan().rename(self._rename_map).select(pl.col("object_id")).unique().collect()
        ids = df["object_id"].to_list()
        return ids[:limit] if limit is not None else ids

    @staticmethod
    def _normalize_bands(filters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (bands, keep_mask): 0 = g, 1 = r; epochs in other passbands
        (e.g. ZTF i-band, fid=3) are dropped via keep_mask.
        """
        if np.issubdtype(filters.dtype, np.number):
            fid = filters.astype(np.int64)
            keep = np.isin(fid, (1, 2))
            return np.where(fid == 2, 1, 0), keep
        low = np.char.lower(filters.astype(str))
        is_g = np.char.find(low, "g") >= 0
        is_r = np.char.find(low, "r") >= 0
        keep = is_g ^ is_r  # exactly one of g/r matched
        return np.where(is_r, 1, 0), keep

    def stream_objects(self, object_ids: Optional[List[Union[int, str]]] = None):
        """Yields preprocessed per-object light-curve records."""
        df = self._load(object_ids)
        has_coords = "ra" in df.columns and "dec" in df.columns

        for obj_id, group in df.group_by("object_id", maintain_order=True):
            filters = group["filter"].to_numpy()
            bands_all, keep = self._normalize_bands(filters)
            if keep.sum() < self.min_obs:
                continue

            mjd = group["mjd"].to_numpy().astype(np.float64)[keep]
            mag = group["mag"].to_numpy().astype(np.float64)[keep]
            magerr = group["magerr"].to_numpy().astype(np.float64)[keep]
            bands = bands_all[keep]

            # Drop non-finite photometry defensively (real archives have NaNs).
            finite = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(magerr)
            if finite.sum() < self.min_obs:
                continue
            mjd, mag, magerr, bands = mjd[finite], mag[finite], magerr[finite], bands[finite]

            t_days = mjd - mjd[0]
            scaler = RobustScalerPerBand()
            norm_mag, norm_magerr = scaler.fit_transform(mag, magerr, bands)

            rec = {
                "object_id": str(obj_id[0] if isinstance(obj_id, tuple) else obj_id),
                "t": t_days.astype(np.float32),
                "mag": norm_mag,
                "magerr": norm_magerr,
                "band": bands.astype(np.int64),
                "raw_mag": mag.astype(np.float32),
                "raw_magerr": magerr.astype(np.float32),
                "raw_t": mjd.astype(np.float32),
                "label": np.int64(0),  # unknown; real data has no truth labels
            }
            if has_coords:
                ra_all = group["ra"].to_numpy().astype(np.float64)
                dec_all = group["dec"].to_numpy().astype(np.float64)
                rec["ra"] = ra_all[keep][finite]
                rec["dec"] = dec_all[keep][finite]
            yield rec

    def load_records(
        self, limit: Optional[int] = None, object_ids: Optional[List] = None
    ) -> List[Dict]:
        """Materializes up to `limit` objects into memory."""
        if object_ids is None and limit is not None:
            object_ids = self.get_object_ids(limit=limit)
        return list(self.stream_objects(object_ids))


class LightCurveDataset(Dataset):
    """
    Windows/pads variable-length light curves into fixed model tensors.

    Output per item:
        x         [2, seq_len]  per-band normalized magnitudes (g row / r row;
                                zeros where the band was not observed)
        t         [seq_len]     epochs scaled to [0, 1] over the window
        obs_mask  [seq_len]     1 where a real observation exists, 0 = padding
        band_mask [seq_len]     0 = g, 1 = r (undefined where obs_mask = 0)
        label     int           ground-truth label if available (validation only)
    """

    def __init__(
        self,
        records: List[Dict[str, np.ndarray]],
        seq_len: int = 256,
        feature_mode: str = "multichannel",
        train: bool = True,
        seed: int = 0,
    ):
        self.records = records
        self.seq_len = seq_len
        self.feature_mode = feature_mode
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        t = np.asarray(rec["t"], dtype=np.float32)
        mag = np.asarray(rec["mag"], dtype=np.float32)
        magerr = np.asarray(rec["magerr"], dtype=np.float32)
        band = np.asarray(rec["band"], dtype=np.int64)
        n_obs = len(t)

        if n_obs > self.seq_len:
            # Random window during training; deterministic leading window for
            # validation/inference so metrics are reproducible.
            start = int(self.rng.integers(0, n_obs - self.seq_len + 1)) if self.train else 0
            sl = slice(start, start + self.seq_len)
            t, mag, magerr, band = t[sl], mag[sl], magerr[sl], band[sl]
            obs_mask = np.ones(self.seq_len, dtype=np.float32)
        else:
            pad = self.seq_len - n_obs
            t = np.pad(t, (0, pad))
            mag = np.pad(mag, (0, pad))
            magerr = np.pad(magerr, (0, pad))
            band = np.pad(band, (0, pad))
            obs_mask = np.pad(np.ones(n_obs, dtype=np.float32), (0, pad))

        # Scale epochs to [0, 1] over the observed window; keeps the time-
        # embedding MLP in a well-conditioned input range regardless of
        # whether callers pass days-since-t0 or raw MJD.
        observed = obs_mask > 0
        if np.any(observed):
            t0 = float(t[observed].min())
            span = float(t[observed].max()) - t0
            t_scaled = np.where(observed, (t - t0) / (span + 1e-6), 0.0).astype(np.float32)
        else:
            t_scaled = np.zeros_like(t)

        if self.feature_mode == "multichannel":
            x = np.zeros((2, self.seq_len), dtype=np.float32)
            g_sel = (band == 0) & observed
            r_sel = (band == 1) & observed
            x[0, g_sel] = mag[g_sel]
            x[1, r_sel] = mag[r_sel]
            return {
                "x": torch.from_numpy(x),
                "t": torch.from_numpy(t_scaled),
                "obs_mask": torch.from_numpy(obs_mask),
                "band_mask": torch.from_numpy(band),
                "object_id": str(rec.get("object_id", f"obj_{idx}")),
                "label": int(rec.get("label", 0)),
            }

        # 'time_embedded' mode: dense per-epoch feature rows [seq_len, 4].
        feats = np.stack([t_scaled, mag, magerr, band.astype(np.float32)], axis=-1)
        return {
            "x": torch.from_numpy(feats.astype(np.float32)),
            "obs_mask": torch.from_numpy(obs_mask),
            "object_id": str(rec.get("object_id", f"obj_{idx}")),
            "label": int(rec.get("label", 0)),
        }


def collate_lightcurves(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stacks tensor fields; passes metadata through as lists."""
    out: Dict = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif k == "label":
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = [b[k] for b in batch]
    return out
