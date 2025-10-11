"""
RTEMS Scheduling Policy Classification Example

RTEMS 스케줄링 정책 분류를 위한 머신러닝 파이프라인 예제
- 데이터: 각 experiment의 가변 길이 task 정보 (CA, U)
- 목표: 스케줄링 정책 분류 (0: global, 1: clustered, 2: partitioned)
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    RtemsJsonDataLoader,
    RtemsFeatureEngineeringPreprocessor,
    XGBoostClassifier,
    RandomForestClassifier,
    StandardScalerPreprocessor
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_rtems_xgboost():
    """예시 1: XGBoost를 사용한 RTEMS 스케줄링 정책 분류"""
    print("\n" + "=" * 80)
    print("RTEMS Scheduling Policy Classification with XGBoost")
    print("=" * 80 + "\n")

    # 데이터 경로 설정
    data_path = "/opt/rtems-ml-framework/data/data2.json"

    # 파이프라인 구성
    config = PipelineConfig(
        data_source=data_path,
        train_test_split=0.2,  # 80% train, 20% test
        random_state=42
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(RtemsJsonDataLoader())
        .with_preprocessing(RtemsFeatureEngineeringPreprocessor())
        .with_model(XGBoostClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ))
        .build()
    )

    # 파이프라인 실행
    results = pipeline.run()

    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    return results


def example_rtems_random_forest():
    """예시 2: Random Forest를 사용한 RTEMS 스케줄링 정책 분류"""
    print("\n" + "=" * 80)
    print("RTEMS Scheduling Policy Classification with Random Forest")
    print("=" * 80 + "\n")

    # 데이터 경로 설정
    data_path = "/opt/rtems-ml-framework/data/data2.json"

    # 파이프라인 구성
    config = PipelineConfig(
        data_source=data_path,
        train_test_split=0.2,
        random_state=42
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(RtemsJsonDataLoader())
        .with_preprocessing(RtemsFeatureEngineeringPreprocessor())
        .with_model(RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
        .build()
    )

    # 파이프라인 실행
    results = pipeline.run()

    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    return results


def compare_models():
    """예시 3: 여러 모델 성능 비교"""
    print("\n" + "=" * 80)
    print("Comparing Multiple Models for RTEMS Scheduling Classification")
    print("=" * 80 + "\n")

    # data_path = "data/rtems_experiments.json"
    data_path = "/opt/rtems-ml-framework/data/data (2).json"

    config = PipelineConfig(
        data_source=data_path,
        train_test_split=0.2,
        random_state=42
    )

    models = {
        # 'XGBoost': XGBoostClassifier(n_estimators=100, max_depth=6),
        # 'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10),
        'XGBoost': XGBoostClassifier(),
        'RandomForest': RandomForestClassifier(),
    }

    results_comparison = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 40}")
        print(f"Training {model_name}...")
        print('=' * 40)

        pipeline = (
            MLPipelineBuilder(config)
            .with_data_loader(RtemsJsonDataLoader())
            .with_preprocessing(RtemsFeatureEngineeringPreprocessor())
            .with_model(model)
            .build()
        )

        results = pipeline.run()
        results_comparison[model_name] = results['accuracy']
        print(f"{model_name} Accuracy: {results['accuracy']:.4f}")


        logger.debug("=" * 80)
        logger.debug(f"Final Prediction step with extra data")
        test_data = RtemsJsonDataLoader().load('/opt/rtems-ml-framework/data/test.json')
        processed_data = test_data
        for idx, step in enumerate(pipeline._preprocessing_steps, 1):
            processed_data = step.transform(processed_data)
        if isinstance(processed_data, pd.DataFrame):
            if 'label' in processed_data.columns:
                # X: 피처 컬럼들, y: label 컬럼
                feature_cols = [col for col in processed_data.columns if col not in ['experiment_id', 'label']]
                X = processed_data[feature_cols]
                y = processed_data['label']
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        results['accuracy'] = accuracy

        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        report = classification_report(y, predictions)
        logger.info(f"\n{report}")


        cm = confusion_matrix(y, predictions)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.debug("=" * 80)

    # 최종 비교
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    for model_name, accuracy in sorted(results_comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:20s}: {accuracy:.4f}")

    return results_comparison


if __name__ == "__main__":
    # # 예시 1: XGBoost
    # example_rtems_xgboost()

    # # 예시 2: Random Forest
    # example_rtems_random_forest()

    # 예시 3: 모델 비교
    compare_models()
