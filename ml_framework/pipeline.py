"""
Pipeline Module
머신러닝 파이프라인 핵심 구현 (Product)
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from .protocols import DataLoaderProtocol, PreprocessorProtocol, ModelProtocol, TunerProtocol
from .config import PipelineConfig, TaskType


class MLPipeline:
    """머신러닝 파이프라인 (Builder Pattern의 Product)"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig(data_source="default.csv")
        self._data_loader: Optional[DataLoaderProtocol] = None
        self._preprocessing_steps: List[PreprocessorProtocol] = []
        self._model: Optional[ModelProtocol] = None
        self._tuner: Optional[TunerProtocol] = None
        self._tune_params: Optional[Dict] = None

    @property
    def data_loader(self) -> Optional[DataLoaderProtocol]:
        return self._data_loader

    @data_loader.setter
    def data_loader(self, loader: DataLoaderProtocol) -> None:
        self._data_loader = loader

    @property
    def model(self) -> Optional[ModelProtocol]:
        return self._model

    @model.setter
    def model(self, model: ModelProtocol) -> None:
        self._model = model

    def add_preprocessing_step(self, step: PreprocessorProtocol) -> None:
        """전처리 단계 추가"""
        self._preprocessing_steps.append(step)

    def set_tuner(self, tuner: TunerProtocol, params: Dict) -> None:
        """튜너 설정"""
        self._tuner = tuner
        self._tune_params = params

    def _validate(self) -> None:
        """파이프라인 유효성 검사"""
        if self._data_loader is None:
            raise ValueError("Data loader must be set")
        if self._model is None:
            raise ValueError("Model must be set")

    def run(self) -> Dict[str, Any]:
        """파이프라인 실행"""
        self._validate()

        logger.info("=" * 80)
        logger.info("Starting ML Pipeline Execution")
        logger.info("=" * 80)

        results = {}

        # 1. 데이터 로드
        logger.info(f"Step 1: Loading data from {self.config.data_source}")
        raw_data = self._data_loader.load(self.config.data_source)
        results['raw_data'] = raw_data

        # 2. 전처리
        processed_data = raw_data
        for idx, step in enumerate(self._preprocessing_steps, 1):
            logger.info(f"Step 2.{idx}: Applying preprocessing - {type(step).__name__}")
            processed_data = step.fit_transform(processed_data)
        logger.debug(f"Processed data: \n{processed_data.head()}")
        results['processed_data'] = processed_data

        # 3. 데이터 분할
        logger.info(f"Step 3: Splitting data (train: {1-self.config.train_test_split:.0%}, test: {self.config.train_test_split:.0%})")
        from sklearn.model_selection import train_test_split
        import pandas as pd

        # processed_data가 DataFrame인 경우 X와 y 분리
        if isinstance(processed_data, pd.DataFrame):
            # 작업 타입에 따라 타겟 컬럼 결정
            if self.config.task_type == TaskType.CLASSIFICATION:
                if 'label' in processed_data.columns:
                    # 분류: label 컬럼 사용
                    feature_cols = [col for col in processed_data.columns if col not in ['experiment_id', 'label']]
                    X = processed_data[feature_cols]
                    y = processed_data['label']
                    logger.info(f"Classification task - Features: {len(feature_cols)} columns")
                    logger.debug(f"Training data's target distribution: {y.value_counts().to_dict()}")
                else:
                    raise ValueError("DataFrame must contain 'label' column for classification")
            else:  # REGRESSION
                # 회귀: target_으로 시작하는 컬럼들 또는 지정된 target_columns 사용
                if self.config.target_columns:
                    target_cols = self.config.target_columns
                else:
                    target_cols = [col for col in processed_data.columns if str(col).startswith('target_')]

                if not target_cols:
                    raise ValueError("DataFrame must contain target columns (starting with 'target_') for regression or specify target_columns in config")

                feature_cols = [col for col in processed_data.columns if col not in target_cols]
                X = processed_data[feature_cols]
                y = processed_data[target_cols]
                logger.info(f"Regression task - Features: {len(feature_cols)} columns, Targets: {len(target_cols)} columns")
                logger.info(f"Target columns: {target_cols}")
        else:
            # 다른 형식의 데이터는 사용자가 직접 X, y로 제공해야 함
            raise ValueError("Processed data must be a pandas DataFrame")

        # 분할 시 stratify는 분류에서만 사용
        stratify_param = y if self.config.task_type == TaskType.CLASSIFICATION else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.train_test_split,
            random_state=self.config.random_state,
            shuffle=True,
            stratify=stratify_param
        )
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
        logger.info(f"Train set: {X_train}, \nTest set size: {X_test}")

        # 4. 하이퍼파라미터 튜닝 (설정된 경우)
        if self._tuner and self._tune_params:
            logger.info(f"Step 4: Hyperparameter tuning with {type(self._tuner).__name__}")
            self._model = self._tuner.tune(self._model, self._tune_params, X_train, y_train)
            results['best_params'] = self._tune_params

        # 5. 모델 학습
        logger.info(f"Step 5: Training model - {type(self._model).__name__}")
        self._model.fit(X_train, y_train)

        # 6. 예측 및 평가
        logger.info("Step 6: Making predictions and evaluating")
        predictions = self._model.predict(X_test)
        results['predictions'] = predictions
        results['y_test'] = y_test
        logger.debug(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")

        # 작업 타입에 따른 성능 평가
        if self.config.task_type == TaskType.CLASSIFICATION:
            # 분류 평가 메트릭
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

            accuracy = accuracy_score(y_test, predictions)
            results['accuracy'] = accuracy

            logger.info(f"Model Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            report = classification_report(y_test, predictions)
            logger.info(f"\n{report}")

            cm = confusion_matrix(y_test, predictions)
            logger.info(f"\nConfusion Matrix:\n{cm}")

            results['classification_report'] = report
            results['confusion_matrix'] = cm

        else:  # REGRESSION
            # 회귀 평가 메트릭
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np

            # 전체 평가
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results['mse'] = mse
            results['rmse'] = rmse
            results['mae'] = mae
            results['r2'] = r2

            logger.info(f"\nOverall Regression Metrics:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE:  {mae:.4f}")
            logger.info(f"  R²:   {r2:.4f}")

            # 타겟별 평가 (다중 출력인 경우)
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                logger.info(f"\nPer-target Metrics:")
                for i, target_name in enumerate(y_test.columns):
                    y_true = y_test.iloc[:, i].values
                    y_pred = predictions[:, i]
                    target_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    target_mae = mean_absolute_error(y_true, y_pred)
                    target_r2 = r2_score(y_true, y_pred)
                    logger.info(f"  {target_name}: RMSE={target_rmse:.4f}, MAE={target_mae:.4f}, R²={target_r2:.4f}")

        results['model'] = self._model

        logger.info("=" * 80)
        logger.info("Pipeline Execution Completed Successfully!")
        logger.info("=" * 80)

        return results

    def __repr__(self) -> str:
        return (
            f"MLPipeline(\n"
            f"  data_loader={type(self._data_loader).__name__ if self._data_loader else None},\n"
            f"  preprocessing_steps={[type(s).__name__ for s in self._preprocessing_steps]},\n"
            f"  model={self._model},\n"
            f"  tuner={type(self._tuner).__name__ if self._tuner else None}\n"
            f")"
        )