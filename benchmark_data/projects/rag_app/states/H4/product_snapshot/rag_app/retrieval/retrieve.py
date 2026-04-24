def retrieve_mode() -> str:
    from rag_app.config import RETRIEVER_BACKEND
    return RETRIEVER_BACKEND
