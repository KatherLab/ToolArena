import pytest
from pytest_lazy_fixtures import lf
from tasks.utils import initialize, parametrize_invocation
from toolarena.run import ToolRunResult

import pandas as pd
import os

initialize()


@parametrize_invocation("s2", "s3", "s4")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@pytest.mark.parametrize(
    "invocation_name,invocation",
    [
        ("s2", lf("s2")),
        ("s3", lf("s3")),
        ("s4", lf("s4")),
    ],
)
def test_shape_and_type(invocation: ToolRunResult, invocation_name):
    assert "predictions_file" in invocation.result
    predictions_path = invocation.output_dir / f"preds_{invocation_name}.csv"
    assert "predictions_file" in invocation.result

    df = pd.read_csv(predictions_path)
    assert not df.empty
    assert df.shape[1] == 12  # sample_id + 10 folds + final

@pytest.mark.parametrize(
    "invocation_name,invocation",
    [
        ("s2", lf("s2")),
        ("s3", lf("s3")),
        ("s4", lf("s4")),
    ],
)
def test_prediction_classes(invocation: ToolRunResult, invocation_name):
    # Load prediction CSV
    pred_path = invocation.output_dir / f"preds_{invocation_name}.csv"
    gt_path = invocation.output_dir.parent / "input" / "predictions" / f"preds_{invocation_name}.csv"

    assert pred_path.exists(), f"Missing: {pred_path}"
    assert gt_path.exists(), f"Missing: {gt_path}"

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    # Make sure both have the same shape
    assert pred_df.shape == gt_df.shape, f"CSV shape mismatch: {pred_df.shape} != {gt_df.shape}"

    # Compare each row (ignoring column names)
    for i in range(len(pred_df)):
        pred_row = pred_df.iloc[i].tolist()
        gt_row = gt_df.iloc[i].tolist()
        assert pred_row == gt_row, (
            f"Mismatch in row {i} for invocation {invocation_name}:\n"
            f"Predicted: {pred_row}\nExpected:  {gt_row}"
        )