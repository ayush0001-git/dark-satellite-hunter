"""
Data Engineering & Synthetic Injection Module for Dark Satellite Hunter.
"""
from .synthetic_debris import tumbling_glint_model, inject_debris_into_lightcurve
from .ztf_dataset import PolarsZTFDataset, LightCurveDataset, collate_lightcurves
from .mock_generator import generate_mock_ztf_dataset
