import workflow_app.config


def test_contract() -> None:
    assert workflow_app.config.TASK_STYLE == 'async_compatible'
    assert workflow_app.config.RUNTIME_MODE == 'prefect3'
