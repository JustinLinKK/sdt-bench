import rag_app.config


def test_contract() -> None:
    assert rag_app.config.PIPELINE_STYLE == 'components'
    assert rag_app.config.SERVICE_MODE == 'cli'
