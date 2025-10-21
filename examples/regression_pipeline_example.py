"""
MLPipelineBuilder를 사용한 회귀 모델 예제

프레임워크의 핵심인 MLPipelineBuilder를 사용하여 회귀 모델 파이프라인을 구성합니다.
- Strategy Pattern + Builder Pattern 활용
- NeuralNetworkRegressor와 VGGRegressor 모두 지원
- 일관된 인터페이스로 데이터 로드, 전처리, 학습, 평가
"""

from pathlib import Path
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    TaskType,
    MemoryPatternRegressionDataLoader,
    MemoryPatternPaddingPreprocessor,
    NeuralNetworkRegressor,
    VGGRegressor,
)


def example_1_neural_network_regression():
    """예시 1: NeuralNetworkRegressor를 사용한 기본 회귀 파이프라인"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Neural Network Regression Pipeline")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source=project_root / "data" / "memory_pattern_large.json",
        train_test_split=0.2,
        random_state=42,
        task_type=TaskType.REGRESSION  # 회귀 작업 지정
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(MemoryPatternRegressionDataLoader())
        .with_preprocessing(
            MemoryPatternPaddingPreprocessor(padding_value=-999.0)
        )
        .with_model(
            NeuralNetworkRegressor(
                hidden_layers=[256, 128, 64, 32],
                activation='relu',
                output_dim=6,
                learning_rate=0.001,
                epochs=200,
                batch_size=16,
                validation_split=0.2,
                verbose=1,
                early_stopping_patience=30
            )
        )
        .build()
    )

    results = pipeline.run()

    print("\n" + "-"*80)
    print(f"Final RMSE: {results['rmse']:.4f}")
    print(f"Final R²: {results['r2']:.4f}")
    print("-"*80)

    return results


def example_2_vgg_regression():
    """예시 2: VGGRegressor를 사용한 깊은 신경망 회귀 파이프라인"""
    print("\n" + "="*80)
    print("EXAMPLE 2: VGG-style Deep Network Regression Pipeline")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source=project_root / "data" / "memory_pattern_large.json",
        train_test_split=0.2,
        random_state=42,
        task_type=TaskType.REGRESSION
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(MemoryPatternRegressionDataLoader())
        .with_preprocessing(
            MemoryPatternPaddingPreprocessor(padding_value=-999.0)
        )
        .with_model(
            VGGRegressor(
                architecture='VGG-11',  # VGG-11 프리셋 사용
                output_dim=6,
                learning_rate=0.001,
                epochs=200,
                batch_size=16,
                validation_split=0.2,
                verbose=1,
                early_stopping_patience=30,
                dropout_rate=0.3,
                weight_decay=0.01
            )
        )
        .build()
    )

    results = pipeline.run()

    print("\n" + "-"*80)
    print(f"Final RMSE: {results['rmse']:.4f}")
    print(f"Final R²: {results['r2']:.4f}")
    print("-"*80)

    return results


def example_3_vgg16_custom():
    """예시 3: VGG-16 아키텍처를 사용한 더 깊은 네트워크"""
    print("\n" + "="*80)
    print("EXAMPLE 3: VGG-16 Architecture for Deeper Learning")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source=project_root / "data" / "memory_pattern_large.json",
        train_test_split=0.2,
        random_state=42,
        task_type=TaskType.REGRESSION
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(MemoryPatternRegressionDataLoader())
        .with_preprocessing(
            MemoryPatternPaddingPreprocessor(padding_value=-999.0)
        )
        .with_model(
            VGGRegressor(
                architecture='VGG-16',  # 더 깊은 VGG-16
                output_dim=6,
                learning_rate=0.001,
                epochs=200,
                batch_size=16,
                validation_split=0.2,
                verbose=1,
                early_stopping_patience=30
            )
        )
        .build()
    )

    results = pipeline.run()

    print("\n" + "-"*80)
    print(f"Final RMSE: {results['rmse']:.4f}")
    print(f"Final R²: {results['r2']:.4f}")
    print("-"*80)

    return results


def example_4_custom_architecture():
    """예시 4: 커스텀 아키텍처를 사용한 회귀 파이프라인"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Architecture Regression Pipeline")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source=project_root / "data" / "memory_pattern_large.json",
        train_test_split=0.2,
        random_state=42,
        task_type=TaskType.REGRESSION
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(MemoryPatternRegressionDataLoader())
        .with_preprocessing(
            MemoryPatternPaddingPreprocessor(padding_value=-999.0)
        )
        .with_model(
            VGGRegressor(
                hidden_layers=[512, 256, 128, 64, 32, 16],  # 커스텀 구조
                output_dim=6,
                learning_rate=0.001,
                epochs=200,
                batch_size=16,
                validation_split=0.2,
                verbose=0,  # 로그 최소화
                early_stopping_patience=30
            )
        )
        .build()
    )

    results = pipeline.run()

    print("\n" + "-"*80)
    print(f"Final RMSE: {results['rmse']:.4f}")
    print(f"Final R²: {results['r2']:.4f}")
    print("-"*80)

    return results


def example_5_model_persistence():
    """예시 5: 모델 학습 후 저장하는 파이프라인"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Training Pipeline with Model Persistence")
    print("="*80 + "\n")

    config = PipelineConfig(
        data_source=project_root / "data" / "memory_pattern_large.json",
        train_test_split=0.2,
        random_state=42,
        task_type=TaskType.REGRESSION
    )

    pipeline = (
        MLPipelineBuilder(config)
        .with_data_loader(MemoryPatternRegressionDataLoader())
        .with_preprocessing(
            MemoryPatternPaddingPreprocessor(padding_value=-999.0)
        )
        .with_model(
            VGGRegressor(
                architecture='VGG-11',
                output_dim=6,
                learning_rate=0.001,
                epochs=150,
                batch_size=16,
                validation_split=0.2,
                verbose=1,
                early_stopping_patience=30
            )
        )
        .build()
    )

    results = pipeline.run()

    # 모델 저장
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    model = results['model']
    checkpoint_path = checkpoint_dir / "pipeline_vgg_model.pth"
    model.save(str(checkpoint_path))

    print("\n" + "-"*80)
    print(f"Model saved to: {checkpoint_path}")
    print(f"Final RMSE: {results['rmse']:.4f}")
    print(f"Final R²: {results['r2']:.4f}")
    print("-"*80)

    # 저장된 모델 로드 테스트
    print("\nLoading saved model...")
    loaded_model = VGGRegressor.from_checkpoint(str(checkpoint_path))
    print(f"Model loaded successfully: {loaded_model.architecture}")

    return results


def compare_architectures():
    """보너스: 여러 아키텍처 비교"""
    print("\n" + "="*80)
    print("BONUS: Comparing Different Architectures")
    print("="*80 + "\n")

    results_comparison = {}

    architectures = [
        ('NeuralNetwork', NeuralNetworkRegressor(
            hidden_layers=[256, 128, 64, 32],
            output_dim=6,
            epochs=100,
            batch_size=16,
            verbose=0
        )),
        ('VGG-11', VGGRegressor(
            architecture='VGG-11',
            output_dim=6,
            epochs=100,
            batch_size=16,
            verbose=0
        )),
        ('VGG-16', VGGRegressor(
            architecture='VGG-16',
            output_dim=6,
            epochs=100,
            batch_size=16,
            verbose=0
        )),
    ]

    for name, model in architectures:
        print(f"\nTraining {name}...")

        config = PipelineConfig(
            data_source=project_root / "data" / "memory_pattern_large.json",
            train_test_split=0.2,
            random_state=42,
            task_type=TaskType.REGRESSION
        )

        pipeline = (
            MLPipelineBuilder(config)
            .with_data_loader(MemoryPatternRegressionDataLoader())
            .with_preprocessing(MemoryPatternPaddingPreprocessor(padding_value=-999.0))
            .with_model(model)
            .build()
        )

        results = pipeline.run()
        results_comparison[name] = {
            'rmse': results['rmse'],
            'mae': results['mae'],
            'r2': results['r2']
        }

    # 결과 비교
    print("\n" + "="*80)
    print("Architecture Comparison Results")
    print("="*80)
    print(f"{'Architecture':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*80)
    for arch, metrics in results_comparison.items():
        print(f"{arch:<20} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} {metrics['r2']:<12.4f}")
    print("="*80)

    return results_comparison


if __name__ == "__main__":
    # 예시 1: 기본 신경망 회귀
    example_1_neural_network_regression()

    # 예시 2: VGG-11 회귀
    example_2_vgg_regression()

    # 예시 3: VGG-16 (더 깊은 네트워크)
    # example_3_vgg16_custom()

    # 예시 4: 커스텀 아키텍처
    # example_4_custom_architecture()

    # 예시 5: 모델 저장
    # example_5_model_persistence()

    # 보너스: 아키텍처 비교
    # compare_architectures()