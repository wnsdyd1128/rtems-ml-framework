"""
Configuration Module
파이프라인 설정을 관리하는 데이터 클래스
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum


class TaskType(Enum):
    """파이프라인 작업 타입"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class PipelineConfig:
    """파이프라인 실행을 위한 설정"""

    data_source: str | Path
    train_test_split: float = 0.2
    random_state: int = 42
    verbose: bool = True
    task_type: TaskType = TaskType.CLASSIFICATION  # 작업 타입 (분류 또는 회귀)
    target_columns: Optional[list] = None  # 회귀 작업에서 타겟 컬럼 지정 (None이면 자동 감지)

    def __post_init__(self):
        """설정 유효성 검사"""
        if not 0 < self.train_test_split < 1:
            raise ValueError(f"train_test_split must be between 0 and 1, got {self.train_test_split}")

        if self.random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {self.random_state}")

        # task_type이 문자열로 전달된 경우 Enum으로 변환
        if isinstance(self.task_type, str):
            try:
                self.task_type = TaskType(self.task_type)
            except ValueError:
                raise ValueError(f"task_type must be 'classification' or 'regression', got {self.task_type}")


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