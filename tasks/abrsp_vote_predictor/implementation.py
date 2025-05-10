def abrsp_vote_predictor(
    dataset_csv_file: str = "/mount/input/dataset_csv/myscore_s1.csv",
    label_col: str = "my_score",
    dataset_splits_dir: str = "/mount/input/dataset_splits/s1_100",
    checkpoints_dir: str = "/mount/input/10_fold_checkpoints",
    features_dir: str = "/mount/input/TCGA-LIHC-cTransPath-features-20x",
    fixed_thresholds: list = [
        6.32,
        5.14,
        6.79,
        5.88,
        6.01,
        5.49,
        6.27,
        5.62,
        6.66,
        5.36,
    ],
    output_predictions_file: str = "/mount/output/predictions.csv",
) -> dict:
    """
    Predict binary sensitivity (High vs Low) to atezolizumab-bevacizumab with the 10-fold ABRS-P model: for each sample keep the ordered list of ten raw scores, apply the matching thresholds, count “High” votes, and assign High if ≥ 5, else Low (ties impossible).

    Args:
        dataset_csv_file: CSV that was used to train the checkpoints, listing every slide/patient to evaluate; must include `slide_id` and the column named by `label_col` (values are loaded but not used for prediction).
        label_col: Name of the expression column in `dataset_csv_file`, required only to satisfy the dataset API.
        dataset_splits_dir: Directory with `splits_0.csv` … `splits_9.csv`; rows marked as `test` define which samples are processed for each fold.
        checkpoints_dir: Directory containing the ten fold checkpoints `s_0_checkpoint.pt` … `s_9_checkpoint.pt`.
        features_dir: Folder holding one CTransPath feature tensor (`.pt`) per `slide_id`.
        fixed_thresholds: Ordered list of exactly ten floats used as fold-specific cut-offs to convert continuous scores to High/Low.
        output_predictions_file: Destination CSV with columns `sample_id`, `fold_0` … `fold_9`, and the majority-voted `final`.

    Returns:
        dict with the following structure:
        {
          'predictions': list  # List of dictionaries mirroring the rows of the output CSV (`sample_id`, `fold_0` … `fold_9`, `final`).
        }
    """

    import os
    import sys

    import pandas as pd

    sys.path.append("/workspace/ABRS-P")

    from datasets.dataset_generic_reg import Generic_MIL_Dataset

    from utils.eval_utils_reg import eval

    task_name = "my_task"  # dummy task name

    class Args:
        """Class to mimic argparse.Namespace for holding arguments."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # Prepare args dynamically
    args = Args(
        drop_out=True,
        k=10,
        k_start=-1,
        k_end=10,
        fold=-1,
        label_frac=1,
        data_dir=[features_dir],
        concat_features=False,
        results_dir="/mount/input/",
        eval_dir="",
        splits_dir=dataset_splits_dir,
        split="test",
        models_exp_code=checkpoints_dir,
        save_exp_code="",
        task=task_name,
        norm=None,
        test_stats=False,
        patch_pred=True,
        model_type="clam_mb_reg",
        model_size="small-768",
        model_activation="softplus",
        n_classes=1,
    )
    args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

    print(os.listdir(features_dir))

    dataset = Generic_MIL_Dataset(
        csv_path=dataset_csv_file,
        data_dir=args.data_dir,
        concat_features=args.concat_features,
        shuffle=False,
        print_info=True,
        label_dict={},
        label_col=["my_score"],
        patient_strat=True,
        ignore=[],
    )

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    if args.fold == -1:
        folds = range(start, end)
    else:
        folds = range(args.fold, args.fold + 1)
    ckpt_paths = [
        os.path.join(args.models_dir, "s_{}_checkpoint.pt".format(fold))
        for fold in folds
    ]
    datasets_id = {"train": 0, "val": 1, "test": 2, "all": -1}

    # Dictionary to store predictions per sample per fold
    predictions_by_sample = {}

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        csv_path = os.path.join(args.splits_dir, f"splits_{folds[ckpt_idx]}.csv")
        datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
        split_dataset = datasets[datasets_id[args.split]]

        # Get predictions from eval
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, df = eval(
            split_dataset, args, ckpt_path
        )

        # df[0]: sample_id, df[2]: predicted score (assumption based on original)
        for sample_id, pred_score in zip(df.iloc[:, 0], df.iloc[:, 2]):
            if sample_id not in predictions_by_sample:
                predictions_by_sample[sample_id] = []
            predictions_by_sample[sample_id].append(pred_score)

    # Build final output rows
    output_rows = []
    for sample_id, scores in predictions_by_sample.items():
        # Apply fixed thresholds per fold
        classes = [
            "High" if score > thr else "Low"
            for score, thr in zip(scores, fixed_thresholds)
        ]
        majority_class = "High" if classes.count("High") >= 5 else "Low"
        row = [sample_id] + classes + [majority_class]
        output_rows.append(row)

    # Build DataFrame
    header = ["sample_id"] + [f"fold_{i}" for i in range(len(ckpt_paths))] + ["final"]
    df_out = pd.DataFrame(output_rows, columns=header)

    # Save CSV
    df_out.to_csv(output_predictions_file, index=False)

    # Return result path
    return {"predictions": df_out.to_dict("records")}
