from toy_pkg import greet


def test_toy_greet_stays_stable() -> None:
    assert greet() == "hello from toy"
