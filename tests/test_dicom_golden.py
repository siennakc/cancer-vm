"""Golden-fixture tests for the canonical DICOM loader (T-1.1).

Fixtures are constructed programmatically so the repo carries no binary
medical files; each fixture isolates one correctness trap from the pitfall
register.
"""

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from oncoscope.data.deid import deidentify
from oncoscope.data.dicom_canonical import load_canonical


def _base_dataset(pixels: np.ndarray) -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.PatientID = "TEST123"
    ds.PatientName = "Doe^Jane"
    ds.Modality = "MG"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.1, 0.1]
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    return ds


def test_monochrome2_identity_normalization():
    ramp = np.tile(np.arange(0, 256, dtype=np.uint16), (16, 1)) * 100
    ds = _base_dataset(ramp)
    img = load_canonical(ds)
    assert img.pixels.min() == 0.0 and img.pixels.max() == 1.0
    # brightest pixel should be where raw value was largest
    assert img.pixels[0, -1] > img.pixels[0, 0]


def test_monochrome1_is_inverted():
    ramp = np.tile(np.arange(0, 256, dtype=np.uint16), (16, 1)) * 100
    ds = _base_dataset(ramp)
    ds.PhotometricInterpretation = "MONOCHROME1"
    img = load_canonical(ds)
    # MONOCHROME1: low raw value = bright, so after canonicalization the
    # first column must be brighter than the last.
    assert img.pixels[0, 0] > img.pixels[0, -1]


def test_rescale_slope_intercept_applied():
    flat = np.full((8, 8), 100, dtype=np.uint16)
    flat[0, 0] = 200
    ds = _base_dataset(flat)
    ds.RescaleSlope = 2.0
    ds.RescaleIntercept = -100.0
    img = load_canonical(ds)
    # After rescale: values 100 and 300 -> normalized 0 and 1.
    assert img.pixels[0, 0] == 1.0
    assert img.pixels[1, 1] == 0.0


def test_pixel_spacing_propagates():
    ds = _base_dataset(np.zeros((8, 8), dtype=np.uint16))
    ds.PixelData = (np.arange(64, dtype=np.uint16)).tobytes()
    img = load_canonical(ds)
    assert img.pixel_spacing_mm == (0.1, 0.1)


def test_deidentify_strips_phi_and_private_tags():
    ds = _base_dataset(np.zeros((8, 8), dtype=np.uint16))
    ds.PixelData = (np.arange(64, dtype=np.uint16)).tobytes()
    ds.InstitutionName = "Some Hospital"          # not allowlisted
    ds.add_new(0x00091001, "LO", "vendor secret")  # private tag
    out = deidentify(ds, pseudo_patient_id="P0001")
    assert out.PatientID == "P0001"
    assert str(out.PatientName) == ""
    assert "InstitutionName" not in out
    assert 0x00091001 not in out
    assert "PixelData" in out  # image survives
