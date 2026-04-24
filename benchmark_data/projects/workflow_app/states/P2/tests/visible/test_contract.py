import workflow_app.config


def test_contract() -> None:
    assert workflow_app.config.RUNTIME_MODE == 'prefect3'
    assert workflow_app.config.PARAM_MODEL == 'v2'
