# -*- coding: utf-8 -*-
"""LightGBM import shim.

Native `lightgbm` first. If the wheel/DLL fails (common off Python 3.11.2),
expose sklearn HistGradientBoosting with the LGBMClassifier / early_stopping /
log_evaluation surface used by full_pipeline.py:510+.
"""
from __future__ import annotations

USING_FALLBACK = False

try:
    import lightgbm as lgb  # noqa: F401

    LGBMClassifier = lgb.LGBMClassifier
    LGBMRegressor = getattr(lgb, "LGBMRegressor", None)
    early_stopping = lgb.early_stopping
    log_evaluation = lgb.log_evaluation
except Exception:  # ImportError, OSError (DLL), ValueError
    USING_FALLBACK = True
    from sklearn.ensemble import HistGradientBoostingClassifier

    def early_stopping(stopping_rounds, verbose=False, **_kwargs):
        return None

    def log_evaluation(period=1, **_kwargs):
        return None

    class LGBMClassifier(HistGradientBoostingClassifier):
        """sklearn stand-in. Extra LightGBM kwargs are ignored."""

        def __init__(
            self,
            objective="binary",
            metric=None,
            verbosity=-1,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
            is_unbalance=True,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=None,
            num_leaves=31,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            min_child_samples=20,
            **kwargs
        ):
            depth = max_depth if max_depth not in (None, -1) else None
            super().__init__(
                max_iter=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=depth,
                max_leaf_nodes=int(num_leaves) if num_leaves else 31,
                l2_regularization=float(reg_lambda),
                min_samples_leaf=int(min_child_samples),
                random_state=random_state,
            )

        def fit(self, X, y, eval_set=None, callbacks=None, **kwargs):
            return super().fit(X, y)

    class _LgbModule:
        LGBMClassifier = LGBMClassifier
        early_stopping = staticmethod(early_stopping)
        log_evaluation = staticmethod(log_evaluation)

    lgb = _LgbModule()
    LGBMRegressor = None
