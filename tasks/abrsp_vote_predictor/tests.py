from pathlib import Path

import pandas as pd
import pytest
from pytest_lazy_fixtures import lf
from tasks.utils import initialize, parametrize_invocation
from toolarena.run import ToolRunResult

initialize()


@parametrize_invocation("s2", "s3", "s4")
def test_status(invocation: ToolRunResult):
    assert invocation.status == "success"


@parametrize_invocation("s2", "s3", "s4")
def test_shape_and_type(invocation: ToolRunResult):
    predictions_path = invocation.output_dir / "predictions.csv"
    assert predictions_path.exists(), f"Missing: {predictions_path}"
    df = pd.read_csv(predictions_path)
    assert not df.empty
    assert df.shape[1] == 12  # sample_id + 10 folds + final


@parametrize_invocation("s2", "s3", "s4")
def test_predictions_file_matches_output(invocation: ToolRunResult):
    predictions_path = invocation.output_dir / "predictions.csv"
    df = pd.read_csv(predictions_path)
    predictions = invocation.result["predictions"]
    assert len(predictions) == len(df)
    for i in range(len(predictions)):
        for key in predictions[i].keys():
            assert predictions[i][key] == df.iloc[i][key], (
                f"Mismatch in row {i} for key {key}: {predictions[i][key]} != {df.iloc[i][key]}"
            )


@pytest.mark.parametrize(
    "invocation,expected_predictions_file",
    [
        (
            lf(task),
            Path(__file__).parent.joinpath(
                "data", "tests", "predictions", f"preds_{task}.csv"
            ),
        )
        for task in ("s2", "s3", "s4")
    ],
)
def test_prediction_classes(
    invocation: ToolRunResult,
    expected_predictions_file: Path,
):
    # Load prediction CSV
    pred_path = invocation.output_dir / "predictions.csv"

    assert pred_path.exists(), f"Missing: {pred_path}"
    assert expected_predictions_file.exists(), f"Missing: {expected_predictions_file}"

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(expected_predictions_file)

    # Make sure both have the same shape
    assert pred_df.shape == gt_df.shape, (
        f"CSV shape mismatch: {pred_df.shape} != {gt_df.shape}"
    )

    # Compare each row (ignoring column names)
    for i in range(len(pred_df)):
        pred_row = pred_df.iloc[i].tolist()
        gt_row = gt_df.iloc[i].tolist()
        assert pred_row == gt_row, (
            f"Mismatch in row {i}:\nPredicted: {pred_row}\nExpected:  {gt_row}"
        )
