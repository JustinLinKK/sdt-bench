import workflow_app.config


def test_contract() -> None:
    assert workflow_app.config.DEPLOYMENT_MODE == 'worker'
    assert workflow_app.config.PARAM_MODEL == 'v1'
