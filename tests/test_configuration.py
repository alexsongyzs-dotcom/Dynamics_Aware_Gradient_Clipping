"""Configuration composition tests."""

from src.configuration import deep_merge, parse_override, set_dotted


def test_deep_merge_preserves_unmodified_nested_values() -> None:
    base = {"optimizer": {"name": "sgd", "lr": 0.1}, "seed": 0}
    result = deep_merge(base, {"optimizer": {"lr": 0.2}})
    assert result == {"optimizer": {"name": "sgd", "lr": 0.2}, "seed": 0}
    assert base["optimizer"]["lr"] == 0.1


def test_dotted_override_uses_yaml_scalar_types() -> None:
    config = {"optimizer": {"lr": 0.1}}
    key, value = parse_override("optimizer.lr=0.4")
    set_dotted(config, key, value)
    assert config["optimizer"]["lr"] == 0.4
