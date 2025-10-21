"""
메모리 패턴 기반 성능 예측 회귀 모델 예제

가변 길이의 메모리 패턴 데이터로부터 6개의 성능 메트릭을 예측하는 회귀 모델.
- 입력: 가변 길이 tasks (각 task는 memory_pattern과 ca를 포함)
- 출력: g/c/p 각각의 execution_time, turnaround_time (총 6개 값)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from ml_framework import (
    MemoryPatternRegressionDataLoader,
    MemoryPatternPaddingPreprocessor,
    NeuralNetworkRegressor
)
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    """메모리 패턴 회귀 파이프라인 실행"""

    logger.info("=" * 80)
    logger.info("Memory Pattern Performance Regression Pipeline")
    logger.info("=" * 80)

    # 1. 데이터 로드
    logger.info("Step 1: Loading data")
    # 더 큰 데이터셋 사용 (100 샘플)
    data_path = project_root / "data" / "memory_pattern_large.json"
    loader = MemoryPatternRegressionDataLoader()
    data = loader.load(data_path)

    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Sample data structure:\n{data.head()}")

    # 2. 전처리 (패딩)
    logger.info("\nStep 2: Preprocessing with padding")
    preprocessor = MemoryPatternPaddingPreprocessor(
        max_tasks=None,  # 자동으로 최대값 감지
        max_memory_entries=None,  # 자동으로 최대값 감지
        padding_value=-999.0  # 실제 0과 구분하기 위한 패딩 값
    )

    processed_data = preprocessor.fit_transform(data)
    logger.info(f"Processed data shape: {processed_data.shape}")
    logger.info(f"Columns: {processed_data.columns.tolist()}")

    # 3. X와 y 분리
    logger.info("\nStep 3: Separating features and targets")

    # 타겟 컬럼 추출
    target_cols = [col for col in processed_data.columns if str(col).startswith('target')]
    feature_cols = [col for col in processed_data.columns if not str(col).startswith('target_')]

    X = processed_data[feature_cols]
    y = processed_data[target_cols]

    # 타겟 컬럼 이름에서 'target_' 제거
    y.columns = [col.replace('target_', '') for col in y.columns]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Targets shape: {y.shape}")
    logger.info(f"Target columns: {y.columns.tolist()}")

    # 4. Train/Test 분할
    logger.info("\nStep 4: Splitting data (80% train, 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # 5. 모델 학습
    logger.info("\nStep 5: Training Neural Network Regressor")

    # 개선된 하이퍼파라미터
    model = NeuralNetworkRegressor(
        hidden_layers=[256, 128, 64, 32, 16],  # 5-layer 깊은 네트워크 + BatchNorm
        activation='relu',
        output_dim=6,  # g/c/p 각각 2개씩 총 6개
        learning_rate=0.001,  # AdamW 옵티마이저 사용
        epochs=500,
        batch_size=16,  # 100개 샘플에 적합한 배치 크기
        validation_split=0.2,
        verbose=1,
        early_stopping_patience=50
    )

    model.fit(X_train, y_train)

    # 6. 예측 및 평가
    logger.info("\nStep 6: Making predictions and evaluating")
    predictions = model.predict(X_test)

    # 회귀 평가 메트릭
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # 각 타겟별 평가
    logger.info("\n" + "=" * 80)
    logger.info("Performance Metrics for Each Target")
    logger.info("=" * 80)

    for i, target_name in enumerate(y.columns):
        y_true = y_test.iloc[:, i].values
        y_pred = predictions[:, i]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        logger.info(f"\n{target_name}:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")

    # 전체 평균 평가
    overall_mse = mean_squared_error(y_test.values, predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(y_test.values, predictions)
    overall_r2 = r2_score(y_test.values, predictions)

    logger.info("\n" + "=" * 80)
    logger.info("Overall Performance")
    logger.info("=" * 80)
    logger.info(f"Overall RMSE: {overall_rmse:.4f}")
    logger.info(f"Overall MAE:  {overall_mae:.4f}")
    logger.info(f"Overall R²:   {overall_r2:.4f}")

    # 예측 샘플 출력
    logger.info("\n" + "=" * 80)
    logger.info("Sample Predictions vs Actual")
    logger.info("=" * 80)

    # 첫 3개 샘플 비교
    for i in range(min(3, len(y_test))):
        logger.info(f"\nSample {i+1}:")
        for j, target_name in enumerate(y.columns):
            actual = y_test.iloc[i, j]
            predicted = predictions[i, j]
            error = abs(actual - predicted)
            logger.info(f"  {target_name:20s}: Actual={actual:7.2f}, Predicted={predicted:7.2f}, Error={error:6.2f}")

    # 학습 히스토리 출력
    history = model.get_training_history()
    if history:
        logger.info("\n" + "=" * 80)
        logger.info("Training History")
        logger.info("=" * 80)
        logger.info(f"Final training loss:   {history['train_loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Best validation loss:  {min(history['val_loss']):.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=" * 80)

    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'predictions': predictions,
        'preprocessor': preprocessor,
        'metrics': {
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2
        }
    }


if __name__ == "__main__":
    results = main()
