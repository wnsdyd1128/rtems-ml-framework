"""
메모리 패턴 기반 성능 예측 회귀 모델 예제 (MLPipelineBuilder 사용)

MLPipelineBuilder를 활용한 회귀 파이프라인으로 리팩토링:
- Builder Pattern을 통한 일관된 인터페이스
- 자동 데이터 분할 및 평가
- 프레임워크와 완전 통합

입력: 가변 길이 tasks (각 task는 memory_pattern과 ca를 포함)
출력: g/c/p 각각의 execution_time, turnaround_time (총 6개 값)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    TaskType,
    MemoryPatternRegressionDataLoader,
    MemoryPatternPaddingPreprocessor,
    NeuralNetworkRegressor
)


def main():
    """메모리 패턴 회귀 파이프라인 실행"""

    logger.info("=" * 80)
    logger.info("Memory Pattern Performance Regression with MLPipelineBuilder")
    logger.info("=" * 80)

    # 파이프라인 설정
    config = PipelineConfig(
        data_source=project_root / "data" / "memory_pattern_large.json",
        train_test_split=0.2,
        random_state=42,
        task_type=TaskType.REGRESSION  # 회귀 작업 지정
    )

    # MLPipelineBuilder를 사용한 파이프라인 구성
    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(
            MemoryPatternRegressionDataLoader()
        )
        .with_preprocessing(
            MemoryPatternPaddingPreprocessor(
                max_tasks=None,  # 자동으로 최대값 감지
                max_memory_entries=None,  # 자동으로 최대값 감지
                padding_value=-999.0  # 실제 0과 구분하기 위한 패딩 값
            )
        )
        .with_model(
            NeuralNetworkRegressor(
                hidden_layers=[256, 128, 64, 32, 16],  # 5-layer deep network
                activation='relu',
                output_dim=6,  # g/c/p 각각 2개씩 총 6개
                learning_rate=0.001,  # AdamW optimizer
                epochs=500,
                batch_size=16,
                validation_split=0.2,
                verbose=1,
                early_stopping_patience=50
            )
        )
        .build()
    )

    # 파이프라인 실행
    # 데이터 로드, 전처리, 분할, 학습, 평가가 자동으로 수행됩니다
    results = pipeline.run()

    # 최종 결과 출력
    logger.info("\n" + "=" * 80)
    logger.info("Final Results Summary")
    logger.info("=" * 80)
    logger.info(f"Overall RMSE: {results['rmse']:.4f}")
    logger.info(f"Overall MAE:  {results['mae']:.4f}")
    logger.info(f"Overall R²:   {results['r2']:.4f}")
    logger.info("=" * 80)

    # 모델 저장 (선택사항)
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model_path = checkpoint_dir / "memory_pattern_regressor.pth"
    results['model'].save(str(model_path))
    logger.info(f"\nModel saved to: {model_path}")

    return results


if __name__ == "__main__":
    results = main()
