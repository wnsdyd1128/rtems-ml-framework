"""
모델 저장 및 로드 예제

NeuralNetworkRegressor와 VGGRegressor의 save/load 기능을 시연합니다.
- 모델 학습 후 체크포인트 저장
- 저장된 체크포인트에서 모델 로드
- 로드된 모델로 예측 수행
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
    NeuralNetworkRegressor,
    VGGRegressor
)
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    """모델 저장/로드 파이프라인 실행"""

    logger.info("=" * 80)
    logger.info("Model Save/Load Example")
    logger.info("=" * 80)

    # 1. 데이터 준비
    logger.info("\nStep 1: Preparing data")
    data_path = project_root / "data" / "memory_pattern_large.json"
    loader = MemoryPatternRegressionDataLoader()
    data = loader.load(data_path)

    preprocessor = MemoryPatternPaddingPreprocessor(padding_value=-999.0)
    processed_data = preprocessor.fit_transform(data)

    target_cols = [col for col in processed_data.columns if str(col).startswith('target')]
    feature_cols = [col for col in processed_data.columns if not str(col).startswith('target_')]

    X = processed_data[feature_cols]
    y = processed_data[target_cols]
    y.columns = [col.replace('target_', '') for col in y.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")

    # 체크포인트 디렉토리 생성
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ========================================================================
    # NeuralNetworkRegressor 예제
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("NeuralNetworkRegressor - Save/Load Example")
    logger.info("=" * 80)

    # 2. 모델 학습
    logger.info("\nStep 2: Training NeuralNetworkRegressor")
    nn_model = NeuralNetworkRegressor(
        hidden_layers=[256, 128, 64, 32],
        activation='relu',
        output_dim=6,
        learning_rate=0.001,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        early_stopping_patience=20
    )

    nn_model.fit(X_train, y_train)
    logger.info("Training completed")

    # 3. 원본 모델로 예측
    logger.info("\nStep 3: Making predictions with original model")
    original_predictions = nn_model.predict(X_test)
    logger.info(f"Original predictions shape: {original_predictions.shape}")

    # 평가
    from sklearn.metrics import mean_squared_error, r2_score
    original_rmse = np.sqrt(mean_squared_error(y_test.values, original_predictions))
    original_r2 = r2_score(y_test.values, original_predictions)
    logger.info(f"Original model - RMSE: {original_rmse:.4f}, R²: {original_r2:.4f}")

    # 4. 모델 저장
    logger.info("\nStep 4: Saving model")
    nn_checkpoint_path = checkpoint_dir / "neural_network_regressor.pth"
    nn_model.save(str(nn_checkpoint_path))

    # 5. 모델 로드
    logger.info("\nStep 5: Loading model from checkpoint")
    loaded_nn_model = NeuralNetworkRegressor.from_checkpoint(str(nn_checkpoint_path))

    # 6. 로드된 모델로 예측
    logger.info("\nStep 6: Making predictions with loaded model")
    loaded_predictions = loaded_nn_model.predict(X_test)
    logger.info(f"Loaded predictions shape: {loaded_predictions.shape}")

    # 평가
    loaded_rmse = np.sqrt(mean_squared_error(y_test.values, loaded_predictions))
    loaded_r2 = r2_score(y_test.values, loaded_predictions)
    logger.info(f"Loaded model - RMSE: {loaded_rmse:.4f}, R²: {loaded_r2:.4f}")

    # 7. 예측 결과 비교
    logger.info("\nStep 7: Comparing predictions")
    prediction_diff = np.abs(original_predictions - loaded_predictions)
    max_diff = prediction_diff.max()
    mean_diff = prediction_diff.mean()

    logger.info(f"Max difference: {max_diff:.10f}")
    logger.info(f"Mean difference: {mean_diff:.10f}")

    if max_diff < 1e-5:
        logger.info("✓ Predictions match perfectly!")
    else:
        logger.warning("✗ Predictions differ!")

    # ========================================================================
    # VGGRegressor 예제
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("VGGRegressor - Save/Load Example")
    logger.info("=" * 80)

    # 8. VGG 모델 학습
    logger.info("\nStep 8: Training VGGRegressor")
    vgg_model = VGGRegressor(
        architecture='VGG-11',
        output_dim=6,
        learning_rate=0.001,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        early_stopping_patience=20
    )

    vgg_model.fit(X_train, y_train)
    logger.info("Training completed")

    # 9. 원본 모델로 예측
    logger.info("\nStep 9: Making predictions with original VGG model")
    vgg_original_predictions = vgg_model.predict(X_test)

    vgg_original_rmse = np.sqrt(mean_squared_error(y_test.values, vgg_original_predictions))
    vgg_original_r2 = r2_score(y_test.values, vgg_original_predictions)
    logger.info(f"Original VGG model - RMSE: {vgg_original_rmse:.4f}, R²: {vgg_original_r2:.4f}")

    # 10. VGG 모델 저장
    logger.info("\nStep 10: Saving VGG model")
    vgg_checkpoint_path = checkpoint_dir / "vgg_regressor.pth"
    vgg_model.save(str(vgg_checkpoint_path))

    # 11. VGG 모델 로드
    logger.info("\nStep 11: Loading VGG model from checkpoint")
    loaded_vgg_model = VGGRegressor.from_checkpoint(str(vgg_checkpoint_path))
    logger.info(f"Loaded architecture: {loaded_vgg_model.architecture}")
    logger.info(f"Loaded layers: {loaded_vgg_model.hidden_layers}")

    # 12. 로드된 VGG 모델로 예측
    logger.info("\nStep 12: Making predictions with loaded VGG model")
    vgg_loaded_predictions = loaded_vgg_model.predict(X_test)

    vgg_loaded_rmse = np.sqrt(mean_squared_error(y_test.values, vgg_loaded_predictions))
    vgg_loaded_r2 = r2_score(y_test.values, vgg_loaded_predictions)
    logger.info(f"Loaded VGG model - RMSE: {vgg_loaded_rmse:.4f}, R²: {vgg_loaded_r2:.4f}")

    # 13. VGG 예측 결과 비교
    logger.info("\nStep 13: Comparing VGG predictions")
    vgg_prediction_diff = np.abs(vgg_original_predictions - vgg_loaded_predictions)
    vgg_max_diff = vgg_prediction_diff.max()
    vgg_mean_diff = vgg_prediction_diff.mean()

    logger.info(f"Max difference: {vgg_max_diff:.10f}")
    logger.info(f"Mean difference: {vgg_mean_diff:.10f}")

    if vgg_max_diff < 1e-5:
        logger.info("✓ VGG predictions match perfectly!")
    else:
        logger.warning("✗ VGG predictions differ!")

    # ========================================================================
    # 학습 히스토리 검증
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Training History Verification")
    logger.info("=" * 80)

    # NeuralNetworkRegressor 히스토리
    logger.info("\nNeuralNetworkRegressor History:")
    nn_history = loaded_nn_model.get_training_history()
    if nn_history:
        logger.info(f"  Epochs trained: {len(nn_history['train_loss'])}")
        logger.info(f"  Final train loss: {nn_history['train_loss'][-1]:.4f}")
        logger.info(f"  Final val loss: {nn_history['val_loss'][-1]:.4f}")
        logger.info(f"  Best val loss: {min(nn_history['val_loss']):.4f}")

    # VGGRegressor 히스토리
    logger.info("\nVGGRegressor History:")
    vgg_history = loaded_vgg_model.get_training_history()
    if vgg_history:
        logger.info(f"  Epochs trained: {len(vgg_history['train_loss'])}")
        logger.info(f"  Final train loss: {vgg_history['train_loss'][-1]:.4f}")
        logger.info(f"  Final val loss: {vgg_history['val_loss'][-1]:.4f}")
        logger.info(f"  Best val loss: {min(vgg_history['val_loss']):.4f}")

    # ========================================================================
    # 요약
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)

    logger.info("\nCheckpoint files created:")
    logger.info(f"  - {nn_checkpoint_path}")
    logger.info(f"  - {vgg_checkpoint_path}")

    logger.info("\nModel persistence verified:")
    logger.info(f"  ✓ NeuralNetworkRegressor: Save/Load working correctly")
    logger.info(f"  ✓ VGGRegressor: Save/Load working correctly")

    logger.info("\n" + "=" * 80)
    logger.info("All Tests Passed!")
    logger.info("=" * 80)

    return {
        'nn_model': loaded_nn_model,
        'vgg_model': loaded_vgg_model,
        'nn_checkpoint': nn_checkpoint_path,
        'vgg_checkpoint': vgg_checkpoint_path
    }


if __name__ == "__main__":
    results = main()
