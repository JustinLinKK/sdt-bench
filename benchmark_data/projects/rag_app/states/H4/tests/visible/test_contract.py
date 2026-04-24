import rag_app.config


def test_contract() -> None:
    assert rag_app.config.SERVICE_MODE == 'hayhooks'
    assert rag_app.config.RETRIEVER_BACKEND == 'selectable'
