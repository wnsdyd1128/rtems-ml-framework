"""
ML Framework Protocol Definitions
Strategy Pattern을 위한 프로토콜 인터페이스 정의
"""

from typing import Any, Dict, Protocol
from pathlib import Path


class DataLoaderProtocol(Protocol):
    """데이터 로더 프로토콜"""

    def load(self, source: str | Path) -> Any:
        """데이터를 로드하여 반환"""
        ...


class PreprocessorProtocol(Protocol):
    """전처리기 프로토콜"""

    def fit(self, data: Any) -> "PreprocessorProtocol":
        """데이터에 맞춰 전처리기를 학습"""
        ...

    def transform(self, data: Any) -> Any:
        """학습된 전처리기로 데이터 변환"""
        ...

    def fit_transform(self, data: Any) -> Any:
        """학습과 변환을 한번에 수행"""
        ...


class ModelProtocol(Protocol):
    """모델 프로토콜 (scikit-learn API 호환)"""

    def fit(self, X: Any, y: Any) -> "ModelProtocol":
        """모델 학습"""
        ...

    def predict(self, X: Any) -> Any:
        """예측 수행"""
        ...


class TunerProtocol(Protocol):
    """하이퍼파라미터 튜닝 프로토콜"""

    def tune(self, model: ModelProtocol, param_grid: Dict, X: Any, y: Any) -> ModelProtocol:
        """하이퍼파라미터 튜닝 수행 후 최적 모델 반환"""
        ...