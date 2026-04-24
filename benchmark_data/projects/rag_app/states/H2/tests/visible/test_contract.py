import rag_app.config


def test_contract() -> None:
    assert rag_app.config.GENERATOR_MODE == 'builder_generator'
    assert rag_app.config.PIPELINE_STYLE == 'components'
