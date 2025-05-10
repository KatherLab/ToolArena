from pathlib import Path

import numpy as np
from PIL import Image
from tasks.utils import initialize, parametrize_invocation
from toolarena.run import ToolRunResult

initialize()


@parametrize_invocation(
    "single_wsi_low_power", "two_wsi_at_2mpp", "full_dir_1p25_power"
)
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation(
    "single_wsi_low_power", "two_wsi_at_2mpp", "full_dir_1p25_power"
)
def test_num_output_files(invocation: ToolRunResult):
    """Number of generated thumbnails is as expected and matches returned value."""
    out_dir = invocation.output_dir
    pngs = sorted(out_dir.rglob("*.png"))
    assert len(pngs) == invocation.result["num_thumbnails"]


@parametrize_invocation(
    "single_wsi_low_power", "two_wsi_at_2mpp", "full_dir_1p25_power"
)
def test_all_outputs_are_valid_png(invocation: ToolRunResult):
    """Every output file is a valid PNG (checks magic number)."""
    out_dir = invocation.output_dir
    pngs = sorted(out_dir.rglob("*.png"))
    for f in pngs:
        # Check first 8 bytes match the PNG file signature (magic number)
        assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", (
            f"{f.name} is not a valid PNG"
        )


@parametrize_invocation(
    "single_wsi_low_power", "two_wsi_at_2mpp", "full_dir_1p25_power"
)
def test_all_outputs_nonempty(invocation: ToolRunResult):
    """Every output PNG is non-empty (sanity check)."""
    out_dir = invocation.output_dir
    pngs = sorted(out_dir.rglob("*.png"))
    for f in pngs:
        assert f.stat().st_size > 0, f"{f.name} is empty"


@parametrize_invocation("full_dir_1p25_power")
def test_against_ground_truth(invocation: ToolRunResult):
    """Byte-wise equality with committed reference thumbnails."""
    out_dir = invocation.output_dir
    for gt in (
        Path(__file__)
        .parent.joinpath("data", "tests", "ground_truth_thumbs")
        .glob("*.png")
    ):
        produced = next(out_dir.rglob(gt.name), None)  # first match or None
        assert produced is not None, f"{gt.name} missing"
        gt_img = Image.open(gt)
        produced_img = Image.open(produced)
        assert gt_img.size == produced_img.size, f"{gt.name} has different size"
        assert gt_img.mode == produced_img.mode, f"{gt.name} has different mode"
        assert (
            mse := np.mean(
                (
                    np.array(gt_img).astype(np.float32)
                    - np.array(produced_img).astype(np.float32)
                )
                ** 2
            )
        ) < 1e-4, f"{gt.name} differs, mean squared error: {mse}"
