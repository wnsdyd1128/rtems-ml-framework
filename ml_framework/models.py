"""
Model Strategies
다양한 머신러닝 모델 구현
"""

from typing import Any
from loguru import logger


class XGBoostClassifier:
    """XGBoost 분류 모델"""

    def __init__(self, **kwargs):
        from xgboost import XGBClassifier

        self.params = kwargs
        # 기본 파라미터 설정
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        default_params.update(self.params)
        self.params = default_params

        self.model = XGBClassifier(**self.params)
        self._is_fitted = False
        logger.info(f"Initializing XGBoost with params: {self.params}")

    def fit(self, X: Any, y: Any, verbose=True) -> "XGBoostClassifier":
        import pandas as pd

        logger.info("Training XGBoost model")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = X

        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = y

        self.model.fit(X_train, y_train, verbose=verbose)
        self._is_fitted = True
        logger.info("XGBoost training completed")
        return self

    def predict(self, X: Any) -> Any:
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info("Predicting with XGBoost")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        predictions = self.model.predict(X_pred)
        logger.info(f"Predictions shape: {predictions.shape}")
        return predictions

    def predict_proba(self, X: Any) -> Any:
        """예측 확률 반환"""
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        return self.model.predict_proba(X_pred)

    def __repr__(self) -> str:
        return f"XGBoostClassifier(params={self.params})"


class RandomForestClassifier:
    """랜덤 포레스트 분류 모델"""

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier as RFClassifier

        self.params = kwargs
        # 기본 파라미터 설정
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
        default_params.update(self.params)
        self.params = default_params

        self.model = RFClassifier(**self.params)
        self._is_fitted = False
        logger.info(f"Initializing RandomForest with params: {self.params}")

    def fit(self, X: Any, y: Any) -> "RandomForestClassifier":
        import pandas as pd

        logger.info("Training RandomForest model")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = X

        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = y

        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("RandomForest training completed")
        return self

    def predict(self, X: Any) -> Any:
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info("Predicting with RandomForest")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        predictions = self.model.predict(X_pred)
        logger.info(f"Predictions shape: {predictions.shape}")
        return predictions

    def predict_proba(self, X: Any) -> Any:
        """예측 확률 반환"""
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        return self.model.predict_proba(X_pred)

    def __repr__(self) -> str:
        return f"RandomForestClassifier(params={self.params})"

