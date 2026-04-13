from __future__ import annotations

import importlib


def test_mixin_modules_import_without_fla():
    # This should succeed even if `fla` isn't installed.
    mod = importlib.import_module("transbench.modules.mixin_modules")
    assert hasattr(mod, "GroupedQuerySelfAttentionMixin")
