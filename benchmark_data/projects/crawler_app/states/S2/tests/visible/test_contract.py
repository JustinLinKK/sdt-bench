import crawler_app.settings


def test_contract() -> None:
    assert crawler_app.settings.REACTOR_MODE == 'asyncio'
    assert crawler_app.settings.START_IMPL == 'async_start'
