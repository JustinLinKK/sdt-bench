import workflow_app.config


def test_contract() -> None:
    assert workflow_app.config.INTEGRATION_BACKEND == 'deployment'
    assert workflow_app.config.TASK_STYLE == 'async_compatible'
