"""
Configuration Module
파이프라인 설정을 관리하는 데이터 클래스
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """파이프라인 실행을 위한 설정"""

    data_source: str | Path
    train_test_split: float = 0.2
    random_state: int = 42
    verbose: bool = True

    def __post_init__(self):
        """설정 유효성 검사"""
        if not 0 < self.train_test_split < 1:
            raise ValueError(f"train_test_split must be between 0 and 1, got {self.train_test_split}")

        if self.random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {self.random_state}")


@dataclass
class ModelConfig:
    """모델 학습을 위한 설정"""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    learning_rate: float = 0.1
    random_state: int = 42

    def to_dict(self):
        """딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() if v is not None}