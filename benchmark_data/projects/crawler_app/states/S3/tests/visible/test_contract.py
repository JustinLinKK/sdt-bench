import crawler_app.settings


def test_contract() -> None:
    assert crawler_app.settings.OFFSITE_MODE == 'allowlist'
    assert crawler_app.settings.MIDDLEWARE_STYLE == 'async'
