import rag_app.config


def test_contract() -> None:
    assert rag_app.config.RETRIEVER_BACKEND == 'selectable'
    assert rag_app.config.GENERATOR_MODE == 'builder_generator'
