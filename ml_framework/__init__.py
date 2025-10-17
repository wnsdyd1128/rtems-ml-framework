"""
ML Framework Package
Strategy Pattern + Builder Pattern을 활용한 모델 독립적 머신러닝 프레임워크
"""

from .protocols import (
    DataLoaderProtocol,
    PreprocessorProtocol,
    ModelProtocol,
    TunerProtocol
)

from .config import PipelineConfig, ModelConfig

from .pipeline import MLPipeline
from .builder import MLPipelineBuilder

from .loaders import (
    CsvDataLoader,
    JsonDataLoader,
    RtemsJsonDataLoader,
    MemoryPatternRegressionDataLoader,
)

from .preprocessors import (
    StandardScalerPreprocessor,
    MinMaxScalerPreprocessor,
    RobustScalerPreprocessor,
    RtemsFeatureEngineeringPreprocessor,
    MemoryPatternPaddingPreprocessor,
)

from .models import (
    XGBoostClassifier,
    RandomForestClassifier,
    NeuralNetworkRegressor,
    VGGRegressor,
)

from .tuners import (
    GridSearchTuner,
    RandomSearchTuner,
    BayesianOptimizationTuner
)

__version__ = "0.1.0"

__all__ = [
    # Protocols
    "DataLoaderProtocol",
    "PreprocessorProtocol",
    "ModelProtocol",
    "TunerProtocol",

    # Config
    "PipelineConfig",
    "ModelConfig",

    # Core
    "MLPipeline",
    "MLPipelineBuilder",

    # Loaders
    "CsvDataLoader",
    "JsonDataLoader",
    "RtemsJsonDataLoader",
    "MemoryPatternRegressionDataLoader",

    # Preprocessors
    "StandardScalerPreprocessor",
    "MinMaxScalerPreprocessor",
    "RobustScalerPreprocessor",
    "RtemsFeatureEngineeringPreprocessor",
    "MemoryPatternPaddingPreprocessor",

    # Models
    "XGBoostClassifier",
    "RandomForestClassifier",
    "NeuralNetworkRegressor",
    "VGGRegressor",

    # Tuners
    "GridSearchTuner",
    "RandomSearchTuner",
    "BayesianOptimizationTuner",
]