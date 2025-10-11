"""
Builder Module
파이프라인을 단계별로 구성하는 빌더 패턴 구현
"""

from typing import Dict, Optional
from loguru import logger

from .protocols import DataLoaderProtocol, PreprocessorProtocol, ModelProtocol, TunerProtocol
from .pipeline import MLPipeline
from .config import PipelineConfig



class MLPipelineBuilder:
    """머신러닝 파이프라인 빌더 (Builder Pattern)"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._pipeline = MLPipeline(config)
        logger.info("Initialized new MLPipelineBuilder")

    def with_data_loader(self, loader: DataLoaderProtocol) -> "MLPipelineBuilder":
        """데이터 로더 설정

        Args:
            loader: DataLoaderProtocol을 구현한 데이터 로더

        Returns:
            self: 체이닝을 위한 빌더 인스턴스
        """
        self._pipeline.data_loader = loader
        logger.info(f"Set data loader: {type(loader).__name__}")
        return self

    def with_preprocessing(self, *steps: PreprocessorProtocol) -> "MLPipelineBuilder":
        """전처리 단계 추가 (여러 개 한번에 가능)

        Args:
            *steps: PreprocessorProtocol을 구현한 전처리기들

        Returns:
            self: 체이닝을 위한 빌더 인스턴스
        """
        for step in steps:
            self._pipeline.add_preprocessing_step(step)
            logger.info(f"Added preprocessing step: {type(step).__name__}")
        return self

    def with_model(self, model: ModelProtocol) -> "MLPipelineBuilder":
        """모델 설정

        Args:
            model: ModelProtocol을 구현한 머신러닝 모델

        Returns:
            self: 체이닝을 위한 빌더 인스턴스
        """
        self._pipeline.model = model
        logger.info(f"Set model: {type(model).__name__}")
        return self

    def with_tuner(self, tuner: TunerProtocol, param_grid: Dict) -> "MLPipelineBuilder":
        """하이퍼파라미터 튜너 설정

        Args:
            tuner: TunerProtocol을 구현한 하이퍼파라미터 튜너
            param_grid: 탐색할 하이퍼파라미터 그리드

        Returns:
            self: 체이닝을 위한 빌더 인스턴스
        """
        self._pipeline.set_tuner(tuner, param_grid)
        logger.info(f"Set tuner: {type(tuner).__name__}")
        return self

    def build(self) -> MLPipeline:
        """파이프라인 빌드 완료

        Returns:
            MLPipeline: 구성이 완료된 파이프라인 인스턴스
        """
        logger.info("Building pipeline...")
        logger.info(f"Pipeline configuration:\n{self._pipeline}")
        return self._pipeline

