# ML Framework

> 전략 패턴(Strategy Pattern)과 빌더 패턴(Builder Pattern)을 활용한 모델 독립적 머신러닝 분류 프레임워크

## 🎯 주요 특징

- **모델 독립적**: 어떤 머신러닝 모델도 쉽게 통합 가능
- **확장성**: 새로운 전처리기, 모델, 튜너를 간단히 추가
- **유연한 구성**: Fluent Interface를 통한 직관적인 파이프라인 구성
- **타입 안정성**: Protocol 기반의 타입 체킹
- **재사용성**: 각 컴포넌트를 독립적으로 테스트 및 재사용 가능

## 📦 의존성 설치

```bash
pip install -r requirements.txt
```

## 🚀 빠른 시작

### 기본 사용법

```python
from ml_framework import (
    MLPipelineBuilder,
    PipelineConfig,
    CsvDataLoader,
    StandardScalerPreprocessor,
    RandomForestClassifier
)

# 설정 정의
config = PipelineConfig(
    data_source="data/train.csv",
    train_test_split=0.2,
    random_state=42
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
```

### 여러 전처리 단계

```python
from ml_framework import (
    RobustScalerPreprocessor,
    MinMaxScalerPreprocessor
)

pipeline = (
    MLPipelineBuilder(config)
    .with_data_loader(JsonDataLoader())
    .with_preprocessing(
        RobustScalerPreprocessor(),
        MinMaxScalerPreprocessor(feature_range=(0, 1))
    )
    .with_model(RandomForestClassifier())
    .build()
)
```

## 🏗️ 아키텍처

### 디렉토리 구조

```
ml_framework/
├── __init__.py          # 패키지 진입점
├── protocols.py         # Protocol 인터페이스 정의
├── config.py            # 설정 클래스
├── pipeline.py          # 파이프라인 핵심 로직
├── builder.py           # 빌더 패턴 구현
├── loaders.py           # 데이터 로더 구현
├── preprocessors.py     # 전처리기 구현
├── models.py            # 모델 구현
└── tuners.py            # 하이퍼파라미터 튜너 구현

examples/
└── basic_usage.py       # 사용 예시

README.md               # 문서
```

### 디자인 패턴

#### Strategy Pattern
각 컴포넌트(데이터 로더, 전처리기, 모델, 튜너)를 독립적인 전략으로 구현하여 런타임에 교체 가능

```python
# 전략을 쉽게 교체
pipeline.with_model(RandomForestClassifier())  # 전략 1
pipeline.with_model(XGBoostClassifier())       # 전략 2
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

```python
import logging
from typing import Any

logger = logging.getLogger(__name__)

class MyCustomClassifier:
    """커스텀 분류 모델"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self._is_fitted = False
        logger.info(f"Initializing MyCustom with params: {self.params}")
    
    def fit(self, X: Any, y: Any) -> "MyCustomClassifier":
        logger.info("Training MyCustom model")
        # 학습 로직
        self._is_fitted = True
        return self
    
    def predict(self, X: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        logger.info("Predicting with MyCustom")
        # 예측 로직
        return predictions
```

### 새로운 전처리기 추가

```python
class MyCustomPreprocessor:
    """커스텀 전처리기"""
    
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, data: Any) -> "MyCustomPreprocessor":
        logger.info("Fitting MyCustomPreprocessor")
        # fit 로직
        self._is_fitted = True
        return self
    
    def transform(self, data: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        logger.info("Transforming with MyCustomPreprocessor")
        # transform 로직
        return transformed_data
    
    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)
```

## 🎓 사용 예시

더 많은 예시는 `examples/basic_usage.py`를 참고하세요.

```python
# 예시 1: 간단한 파이프라인
python examples/basic_usage.py

# 예시 2: GridSearch 튜닝
python examples/basic_usage.py

# 예시 3: 복잡한 전처리
python examples/basic_usage.py

# 예시 4: RTEMS
python examples/rtems_scheduling_classification.py
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

## 📧 연락처

프로젝트 링크: [https://github.com/yourusername/ml-framework](https://github.com/yourusername/ml-framework)