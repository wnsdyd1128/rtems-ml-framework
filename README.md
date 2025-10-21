# RTEMS ML Framework

> 전략 패턴(Strategy Pattern)과 빌더 패턴(Builder Pattern)을 활용한 모델 독립적 머신러닝 프레임워크

RTEMS(Real-Time Executive for Multiprocessor Systems) 환경에서의 머신러닝 워크로드 분석 및 성능 예측을 위한 확장 가능한 프레임워크입니다.

## 🎯 주요 특징

- **분류 & 회귀 지원**: Classification과 Regression 작업을 모두 지원
- **모델 독립적**: 어떤 머신러닝 모델도 쉽게 통합 가능
- **확장성**: 새로운 전처리기, 모델, 튜너를 간단히 추가
- **유연한 구성**: Fluent Interface를 통한 직관적인 파이프라인 구성
- **타입 안정성**: Protocol 기반의 타입 체킹
- **모델 저장/로드**: PyTorch 모델 체크포인트 관리
- **재사용성**: 각 컴포넌트를 독립적으로 테스트 및 재사용 가능

## 📦 의존성 설치

```bash
pip install -r requirements.txt
```

**주요 의존성:**
- `scikit-learn` - 전통적인 머신러닝 모델 및 전처리
- `torch` - 딥러닝 모델 (NeuralNetworkRegressor, VGGRegressor)
- `xgboost` - Gradient Boosting 모델
- `pandas`, `numpy` - 데이터 처리
- `matplotlib` - 시각화

## 🚀 빠른 시작

### 분류 (Classification) 예제

```python
from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    TaskType,
    CsvDataLoader,
    StandardScalerPreprocessor,
    RandomForestClassifier
)

# 설정 정의
config = PipelineConfig(
    data_source="data/train.csv",
    train_test_split=0.2,
    random_state=42,
    task_type=TaskType.CLASSIFICATION  # 분류 작업
)

# 파이프라인 구성
pipeline = (
    MLPipelineBuilder(config)
    .with_data_loader(CsvDataLoader())
    .with_preprocessing(StandardScalerPreprocessor())
    .with_model(RandomForestClassifier(n_estimators=100))
    .build()
)

# 실행
results = pipeline.run()
print(f"Accuracy: {results['accuracy']:.4f}")
```

### 회귀 (Regression) 예제

```python
from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    TaskType,
    MemoryPatternRegressionDataLoader,
    MemoryPatternPaddingPreprocessor,
    NeuralNetworkRegressor
)

config = PipelineConfig(
    data_source="data/memory_pattern_large.json",
    train_test_split=0.2,
    random_state=42,
    task_type=TaskType.REGRESSION  # 회귀 작업
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
            output_dim=6,  # 6개의 타겟 변수
            learning_rate=0.001,
            epochs=300,
            batch_size=16
        )
    )
    .build()
)

results = pipeline.run()
print(f"RMSE: {results['rmse']:.4f}")
print(f"R²: {results['r2']:.4f}")
```

### 모델 저장 및 로드

```python
from ml_framework import NeuralNetworkRegressor

# 1. 모델 학습 및 저장
model = NeuralNetworkRegressor(hidden_layers=[128, 64], output_dim=6)
model.fit(X_train, y_train)
model.save("checkpoint.pth")

# 2. 저장된 모델 로드
loaded_model = NeuralNetworkRegressor.from_checkpoint("checkpoint.pth")
predictions = loaded_model.predict(X_test)
```

### 하이퍼파라미터 튜닝

```python
from ml_framework import (
    XGBoostClassifier,
    GridSearchTuner
)

pipeline = (
    MLPipelineBuilder(config)
    .with_data_loader(CsvDataLoader())
    .with_preprocessing(StandardScalerPreprocessor())
    .with_model(XGBoostClassifier())
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
print(f"Best parameters: {results.get('best_params')}")
```

## 🏗️ 아키텍처

### 디렉토리 구조

```
rtems-ml-framework/
├── ml_framework/
│   ├── __init__.py          # 패키지 진입점
│   ├── protocols.py         # Protocol 인터페이스 정의
│   ├── config.py            # 설정 클래스 (PipelineConfig, TaskType)
│   ├── pipeline.py          # 파이프라인 핵심 로직
│   ├── builder.py           # 빌더 패턴 구현
│   ├── loaders.py           # 데이터 로더 구현
│   ├── preprocessors.py     # 전처리기 구현
│   ├── models.py            # 모델 구현
│   └── tuners.py            # 하이퍼파라미터 튜너 구현
│
├── examples/
│   ├── basic_usage.py                          # 기본 사용 예제
│   ├── rtems_scheduling_classification.py      # RTEMS 스케줄링 분류
│   ├── memory_pattern_regression.py            # 메모리 패턴 회귀 (MLPipelineBuilder)
│   ├── memory_pattern_regression_standalone.py # 메모리 패턴 회귀 (Standalone)
│   ├── regression_pipeline_example.py          # 회귀 파이프라인 예제
│   ├── model_save_load_example.py              # 모델 저장/로드 예제
│   └── vgg_regression_comparison.py            # VGG vs 일반 네트워크 비교
│
├── data/                    # 데이터셋
├── .gitignore
├── requirements.txt
└── README.md
```

### 핵심 컴포넌트

#### 1. 데이터 로더 (Data Loaders)
- `CsvDataLoader` - CSV 파일 로드
- `JsonDataLoader` - JSON 파일 로드
- `RtemsJsonDataLoader` - RTEMS 전용 JSON 데이터 로드
- `MemoryPatternRegressionDataLoader` - 메모리 패턴 회귀 데이터 로드

#### 2. 전처리기 (Preprocessors)
- `StandardScalerPreprocessor` - 표준화 (평균 0, 분산 1)
- `MinMaxScalerPreprocessor` - Min-Max 정규화
- `RobustScalerPreprocessor` - 이상치에 강건한 스케일링
- `RtemsFeatureEngineeringPreprocessor` - RTEMS 특성 공학
- `MemoryPatternPaddingPreprocessor` - 가변 길이 메모리 패턴 패딩

#### 3. 모델 (Models)

**분류 모델:**
- `RandomForestClassifier` - 랜덤 포레스트
- `XGBoostClassifier` - XGBoost

**회귀 모델:**
- `NeuralNetworkRegressor` - PyTorch 기반 다층 신경망 (모델 저장/로드 지원)
- `VGGRegressor` - VGG-style 깊은 신경망 (BatchNorm, Dropout 포함)

#### 4. 튜너 (Tuners)
- `GridSearchTuner` - 그리드 서치
- `RandomSearchTuner` - 랜덤 서치
- `BayesianOptimizationTuner` - 베이지안 최적화

### 디자인 패턴

#### Strategy Pattern
각 컴포넌트(데이터 로더, 전처리기, 모델, 튜너)를 독립적인 전략으로 구현하여 런타임에 교체 가능

```python
# 전략을 쉽게 교체
pipeline.with_model(RandomForestClassifier())  # 전략 1
pipeline.with_model(XGBoostClassifier())       # 전략 2
pipeline.with_model(NeuralNetworkRegressor())  # 전략 3
```

#### Builder Pattern
복잡한 파이프라인을 단계별로 구성하여 가독성과 유지보수성 향상

```python
pipeline = (
    MLPipelineBuilder(config)
    .with_data_loader(...)
    .with_preprocessing(...)
    .with_model(...)
    .build()
)
```

## 🔧 확장하기

### 새로운 모델 추가

Protocol을 구현하여 새로운 모델을 쉽게 추가할 수 있습니다.

```python
from loguru import logger
from typing import Any

class MyCustomRegressor:
    """커스텀 회귀 모델"""

    def __init__(self, **kwargs):
        self.params = kwargs
        self._is_fitted = False
        logger.info(f"Initializing MyCustomRegressor with params: {self.params}")

    def fit(self, X: Any, y: Any) -> "MyCustomRegressor":
        logger.info("Training MyCustomRegressor model")
        # 학습 로직 구현
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        logger.info("Predicting with MyCustomRegressor")
        # 예측 로직 구현
        return predictions

    def save(self, path: str):
        """모델 저장 (선택사항)"""
        pass

    @classmethod
    def from_checkpoint(cls, path: str) -> "MyCustomRegressor":
        """체크포인트에서 모델 로드 (선택사항)"""
        pass
```

### 새로운 전처리기 추가

```python
from loguru import logger
from typing import Any

class MyCustomPreprocessor:
    """커스텀 전처리기"""

    def __init__(self):
        self._is_fitted = False

    def fit(self, data: Any) -> "MyCustomPreprocessor":
        logger.info("Fitting MyCustomPreprocessor")
        # fit 로직 구현
        self._is_fitted = True
        return self

    def transform(self, data: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        logger.info("Transforming with MyCustomPreprocessor")
        # transform 로직 구현
        return transformed_data

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)
```

## 🎓 사용 예시

### 예제 실행

```bash
# 기본 분류 파이프라인
python examples/basic_usage.py

# RTEMS 스케줄링 분류
python examples/rtems_scheduling_classification.py

# 메모리 패턴 회귀 (MLPipelineBuilder 사용)
python examples/memory_pattern_regression.py

# 메모리 패턴 회귀 (Standalone)
python examples/memory_pattern_regression_standalone.py

# 회귀 파이프라인 예제
python examples/regression_pipeline_example.py

# 모델 저장/로드 예제
python examples/model_save_load_example.py

# VGG vs 일반 네트워크 성능 비교
python examples/vgg_regression_comparison.py
```

## 📊 TaskType 설정

프레임워크는 두 가지 작업 타입을 지원합니다:

### Classification (분류)
- 타겟: `label` 컬럼
- 평가 메트릭: Accuracy, F1-Score, Confusion Matrix
- Train/Test 분할 시 stratify 적용

```python
config = PipelineConfig(
    data_source="data.csv",
    task_type=TaskType.CLASSIFICATION
)
```

### Regression (회귀)
- 타겟: `target_*` 컬럼 또는 `config.target_columns` 지정
- 평가 메트릭: RMSE, MAE, R²
- 다중 출력 회귀 지원 (타겟별 메트릭 계산)

```python
config = PipelineConfig(
    data_source="data.csv",
    task_type=TaskType.REGRESSION,
    target_columns=['target_1', 'target_2']  # 선택사항
)
```

## 🚀 고급 기능

### 1. 다중 전처리 단계

```python
pipeline = (
    MLPipelineBuilder(config)
    .with_preprocessing(
        RobustScalerPreprocessor(),
        MinMaxScalerPreprocessor(feature_range=(0, 1))
    )
    .with_model(RandomForestClassifier())
    .build()
)
```

### 2. 모델 체크포인트 관리

```python
# 학습 중 체크포인트 저장
model = NeuralNetworkRegressor(...)
model.fit(X_train, y_train)
model.save("best_model.pth")

# 나중에 로드하여 예측
model = NeuralNetworkRegressor.from_checkpoint("best_model.pth")
predictions = model.predict(X_new)
```

### 3. 가변 길이 데이터 처리

메모리 패턴과 같은 가변 길이 데이터를 자동으로 패딩하여 고정 크기로 변환:

```python
preprocessor = MemoryPatternPaddingPreprocessor(
    max_tasks=None,  # 자동 감지
    max_memory_entries=None,  # 자동 감지
    padding_value=-999.0
)
```

## 📝 라이선스

MIT License

## 🤝 기여하기

기여는 언제나 환영합니다! Pull Request를 보내주세요.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Issue Templates

프로젝트는 다음 이슈 템플릿을 제공합니다:
- `✨Feat` - 새로운 기능 추가
- `🐛Fix` - 버그 수정

## 📧 연락처

프로젝트 링크: [https://github.com/yourusername/rtems-ml-framework](https://github.com/yourusername/rtems-ml-framework)

## 🔗 관련 프로젝트

- [RTEMS](https://www.rtems.org/) - Real-Time Executive for Multiprocessor Systems