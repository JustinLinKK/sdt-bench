import crawler_app.settings


def test_contract() -> None:
    assert crawler_app.settings.REACTOR_MODE == 'sync'
    assert crawler_app.settings.START_IMPL == 'start_requests'
