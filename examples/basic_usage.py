"""
Basic Usage Examples
ML Framework 사용 예시
"""

import logging

from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    CsvDataLoader,
    JsonDataLoader,
    StandardScalerPreprocessor,
    MinMaxScalerPreprocessor,
    RobustScalerPreprocessor,
    XGBoostClassifier,
    RandomForestClassifier,
    GridSearchTuner,
    RandomSearchTuner,
    BayesianOptimizationTuner,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_1_simple_pipeline():
    """예시 1: 기본 파이프라인 (튜닝 없음)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Pipeline without Tuning")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source="data/train.csv",
        train_test_split=0.2,
        random_state=42
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(CsvDataLoader())
        .with_preprocessing(StandardScalerPreprocessor())
        .with_model(RandomForestClassifier(n_estimators=100, max_depth=10))
        .build()
    )

    results = pipeline.run()
    return results


def example_2_with_grid_search():
    """예시 2: GridSearch를 사용한 하이퍼파라미터 튜닝"""
    print("\n" + "="*80)
    print("EXAMPLE 2: XGBoost with GridSearch Tuning")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source="data/train.csv",
        train_test_split=0.2,
        random_state=42
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(CsvDataLoader())
        .with_preprocessing(
            StandardScalerPreprocessor()
        )
        .with_model(XGBoostClassifier(n_estimators=100))
        .with_tuner(
            GridSearchTuner(cv=5, scoring='f1'),
            param_grid={
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        )
        .build()
    )

    results = pipeline.run()
    return results


def example_3_multiple_preprocessing():
    """예시 3: 여러 전처리 단계를 가진 복잡한 파이프라인"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Complex Pipeline with Multiple Preprocessing Steps")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source="data/complex_data.json",
        train_test_split=0.3,
        random_state=123
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(JsonDataLoader())
        .with_preprocessing(
            RobustScalerPreprocessor(),
            MinMaxScalerPreprocessor(feature_range=(0, 1))
        )
        .with_model(LightGBMClassifier(n_estimators=200, max_depth=8))
        .with_tuner(
            RandomSearchTuner(n_iter=20, cv=3, scoring='roc_auc'),
            param_grid={
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [5, 7, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        )
        .build()
    )

    results = pipeline.run()
    return results


def example_4_bayesian_optimization():
    """예시 4: 베이지안 최적화를 사용한 튜닝"""
    print("\n" + "="*80)
    print("EXAMPLE 4: XGBoost with Bayesian Optimization")
    print("="*80 + "\n")

    pipeline = (
        MLPipelineBuilder()
        .with_data_loader(CsvDataLoader())
        .with_preprocessing(StandardScalerPreprocessor())
        .with_model(XGBoostClassifier())
        .with_tuner(
            BayesianOptimizationTuner(n_iter=50, cv=5),
            param_grid={
                'max_depth': [3, 5, 7, 10, 15],
                'learning_rate': [0.001, 0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200, 500]
            }
        )
        .build()
    )

    results = pipeline.run()
    return results


def example_5_minimal():
    """예시 5: 최소 설정으로 빠른 실험"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Minimal Configuration for Quick Experiment")
    print("="*80 + "\n")

    pipeline = (
        MLPipelineBuilder()
        .with_data_loader(CsvDataLoader())
        .with_model(RandomForestClassifier())
        .build()
    )

    results = pipeline.run()
    return results


if __name__ == "__main__":
    # 원하는 예시를 실행

    # 예시 1: 기본 파이프라인
    example_1_simple_pipeline()

    # 예시 2: GridSearch 튜닝
    example_2_with_grid_search()

    # 예시 3: 복잡한 전처리
    example_3_multiple_preprocessing()

    # 예시 4: 베이지안 최적화
    # example_4_bayesian_optimization()

    # 예시 5: 최소 설정
    # example_5_minimal()
