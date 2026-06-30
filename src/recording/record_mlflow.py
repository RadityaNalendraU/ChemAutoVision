from datetime import datetime

import mlflow


def record_exp_result(
    exp_id: int,
    metrics: dict,
    params: dict,
    tags: dict,
) -> None:
    mlflow.set_tracking_uri("../mlruns")
    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{tags['target']}_{tags['model_name']}_{tags['explanatory_val']}",
    ):
        for key, value in tags.items():
            mlflow.set_tag(key=key, value=value)

        for key, value in params.items():
            mlflow.log_param(key=key, value=value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # TODO: artifactで保存できるようにする
        # roc_file_path = create_roc_curve(y_score, y_test, tags["model_name"])
        # breakpoint()
        # mlflow.log_artifact(roc_file_path)
