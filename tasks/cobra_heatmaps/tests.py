import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from tasks.utils import initialize, parametrize_invocation
from toolarena.run import ToolRunResult

initialize()


@parametrize_invocation("crc", "brca", "brca-single")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@pytest.mark.parametrize(
    "invocation,expected_num_heatmaps",
    [
        (lf("crc"), 8),
        (lf("brca"), 9),
        (lf("brca-single"), 1),
    ],
)
def test_num_heatmaps(invocation: ToolRunResult, expected_num_heatmaps: int):
    assert invocation.result["num_heatmaps"] == expected_num_heatmaps


@pytest.mark.parametrize(
    "invocation,expected_byte_size_heatmaps",
    [
        (lf("crc"), 15574480),
        (lf("brca"), 8501263),
        (lf("brca-single"), 1889248),
    ],
)
def test_byte_size_heatmaps(
    invocation: ToolRunResult, expected_byte_size_heatmaps: int
):
    np.testing.assert_almost_equal(
        invocation.result["byte_size_heatmaps"],
        expected_byte_size_heatmaps,
        decimal=-2,
    )


@pytest.mark.parametrize(
    "invocation,expected_num_pdfs",
    [
        (lf("crc"), 8),
        (lf("brca"), 9),
        (lf("brca-single"), 1),
    ],
)
def test_output_files_have_correct_shape_and_type(
    invocation: ToolRunResult, expected_num_pdfs: int
):
    nr_heatmaps = 0
    for pdf_file in invocation.output_dir.rglob("**/*.pdf"):
        with open(pdf_file, "rb") as f:
            assert f.read(4) == b"%PDF"
            nr_heatmaps += 1
    assert nr_heatmaps == expected_num_pdfs
