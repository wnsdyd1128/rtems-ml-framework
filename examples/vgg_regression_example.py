"""
VGG-style 신경망 회귀 모델 예제

VGG 논문의 핵심 아이디어를 1D 회귀 문제에 적용한 예제:
- 깊은 네트워크 구조 (VGG-11, VGG-16, VGG-19)
- Batch Normalization과 Dropout을 통한 정규화
- Xavier 초기화
- 학습률 스케줄링
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
    VGGRegressor
)
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    """VGG 회귀 파이프라인 실행"""

    logger.info("=" * 80)
    logger.info("VGG-style Neural Network Regression Pipeline")
    logger.info("=" * 80)

    # 1. 데이터 로드
    logger.info("Step 1: Loading data")
    data_path = project_root / "data" / "memory_pattern_large.json"
    loader = MemoryPatternRegressionDataLoader()
    data = loader.load(data_path)

    logger.info(f"Loaded {len(data)} samples")

    # 2. 전처리 (패딩)
    logger.info("\nStep 2: Preprocessing with padding")
    preprocessor = MemoryPatternPaddingPreprocessor(
        max_tasks=None,  # 자동으로 최대값 감지
        max_memory_entries=None,  # 자동으로 최대값 감지
        padding_value=-999.0
    )

    processed_data = preprocessor.fit_transform(data)
    logger.info(f"Processed data shape: {processed_data.shape}")

    # 3. X와 y 분리
    logger.info("\nStep 3: Separating features and targets")

    target_cols = [col for col in processed_data.columns if str(col).startswith('target')]
    feature_cols = [col for col in processed_data.columns if not str(col).startswith('target_')]

    X = processed_data[feature_cols]
    y = processed_data[target_cols]

    # 타겟 컬럼 이름에서 'target_' 제거
    y.columns = [col.replace('target_', '') for col in y.columns]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Targets shape: {y.shape}")

    # 4. Train/Test 분할
    logger.info("\nStep 4: Splitting data (80% train, 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # 5. VGG-11 모델 학습
    logger.info("\n" + "=" * 80)
    logger.info("Training VGG-11 Architecture")
    logger.info("=" * 80)

    model_vgg11 = VGGRegressor(
        architecture='VGG-11',  # VGG-11 프리셋 사용
        output_dim=6,
        learning_rate=0.001,
        epochs=500,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        early_stopping_patience=50,
        dropout_rate=0.3,
        weight_decay=0.01
    )

    logger.info(f"Architecture: {model_vgg11.architecture}")
    logger.info(f"Hidden layers: {model_vgg11.hidden_layers}")

    model_vgg11.fit(X_train, y_train)

    # 6. 예측 및 평가
    logger.info("\nStep 6: Making predictions and evaluating")
    predictions = model_vgg11.predict(X_test)

    # 회귀 평가 메트릭
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # 각 타겟별 평가
    logger.info("\n" + "=" * 80)
    logger.info("Performance Metrics for Each Target (VGG-11)")
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
    logger.info("Overall Performance (VGG-11)")
    logger.info("=" * 80)
    logger.info(f"Overall RMSE: {overall_rmse:.4f}")
    logger.info(f"Overall MAE:  {overall_mae:.4f}")
    logger.info(f"Overall R²:   {overall_r2:.4f}")

    # 7. 모델 저장
    logger.info("\nStep 7: Saving model")
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    model_path = checkpoint_dir / "vgg11_regressor.pth"
    model_vgg11.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # 8. 다른 VGG 아키텍처 비교 (선택사항)
    logger.info("\n" + "=" * 80)
    logger.info("Comparing Different VGG Architectures")
    logger.info("=" * 80)

    architectures = ['VGG-11', 'VGG-16']
    results = {}

    for arch in architectures:
        logger.info(f"\n--- Training {arch} ---")

        model = VGGRegressor(
            architecture=arch,
            output_dim=6,
            learning_rate=0.001,
            epochs=300,  # 비교를 위해 에포크 수 줄임
            batch_size=16,
            validation_split=0.2,
            verbose=0,  # 로그 축소
            early_stopping_patience=30,
            dropout_rate=0.3,
            weight_decay=0.01
        )

        logger.info(f"Layers: {len(model.hidden_layers)}, Parameters will be shown during build")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test.values, preds))
        mae = mean_absolute_error(y_test.values, preds)
        r2 = r2_score(y_test.values, preds)

        results[arch] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'layers': len(model.hidden_layers)
        }

        logger.info(f"{arch} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # 결과 비교
    logger.info("\n" + "=" * 80)
    logger.info("Architecture Comparison Summary")
    logger.info("=" * 80)
    logger.info(f"{'Architecture':<15} {'Layers':<8} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    logger.info("-" * 80)
    for arch, metrics in results.items():
        logger.info(f"{arch:<15} {metrics['layers']:<8} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f}")

    # 9. 커스텀 아키텍처 예제
    logger.info("\n" + "=" * 80)
    logger.info("Custom Architecture Example")
    logger.info("=" * 80)

    custom_model = VGGRegressor(
        hidden_layers=[256, 256, 128, 128, 64, 64, 32],  # 커스텀 구조
        output_dim=6,
        learning_rate=0.001,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        early_stopping_patience=30
    )

    logger.info(f"Custom layers: {custom_model.hidden_layers}")
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)

    custom_rmse = np.sqrt(mean_squared_error(y_test.values, custom_preds))
    custom_mae = mean_absolute_error(y_test.values, custom_preds)
    custom_r2 = r2_score(y_test.values, custom_preds)

    logger.info(f"Custom Model Results: RMSE={custom_rmse:.4f}, MAE={custom_mae:.4f}, R²={custom_r2:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=" * 80)

    return {
        'vgg11_model': model_vgg11,
        'results': results,
        'custom_model': custom_model
    }


if __name__ == "__main__":
    results = main()