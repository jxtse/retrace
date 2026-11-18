"""Regression tests for template-library cache invalidation."""

from unittest.mock import Mock

from tools.builtins import template_library


def _seed_caches(monkeypatch, dataset_dir):
    templates = ["template"]
    sources = ["source"]
    reactions = [object()]
    monkeypatch.setattr(template_library, "_DATASET_DIR", dataset_dir.resolve())
    monkeypatch.setattr(template_library, "_ALL_TEMPLATES", templates)
    monkeypatch.setattr(template_library, "_TEMPLATE_SOURCES", sources)
    monkeypatch.setattr(template_library, "_PARSED_REACTIONS", reactions)
    return templates, sources, reactions


def test_same_dataset_dir_preserves_loaded_caches(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "templates"
    dataset_dir.mkdir()
    templates, sources, reactions = _seed_caches(monkeypatch, dataset_dir)
    clear_pattern_cache = Mock()
    monkeypatch.setattr(
        template_library, "_clear_pattern_cache", clear_pattern_cache
    )

    template_library.set_dataset_dir(str(dataset_dir))

    assert template_library._ALL_TEMPLATES is templates
    assert template_library._TEMPLATE_SOURCES is sources
    assert template_library._PARSED_REACTIONS is reactions
    clear_pattern_cache.assert_not_called()


def test_changed_dataset_dir_invalidates_loaded_caches(tmp_path, monkeypatch):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    _seed_caches(monkeypatch, old_dir)
    clear_pattern_cache = Mock()
    monkeypatch.setattr(
        template_library, "_clear_pattern_cache", clear_pattern_cache
    )

    template_library.set_dataset_dir(str(new_dir))

    assert template_library._DATASET_DIR == new_dir.resolve()
    assert template_library._ALL_TEMPLATES is None
    assert template_library._TEMPLATE_SOURCES is None
    assert template_library._PARSED_REACTIONS is None
    clear_pattern_cache.assert_called_once_with()
