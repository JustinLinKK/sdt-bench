import crawler_app.settings


def test_contract() -> None:
    assert crawler_app.settings.EXTENSION_STYLE == 'modern'
    assert crawler_app.settings.MIDDLEWARE_STYLE == 'async'
