"""
Model Strategies
다양한 머신러닝 모델 구현
"""

from typing import Any
from loguru import logger


class XGBoostClassifier:
    """XGBoost 분류 모델"""

    def __init__(self, **kwargs):
        from xgboost import XGBClassifier

        self.params = kwargs
        # 기본 파라미터 설정
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        default_params.update(self.params)
        self.params = default_params

        self.model = XGBClassifier(**self.params)
        self._is_fitted = False
        logger.info(f"Initializing XGBoost with params: {self.params}")

    def fit(self, X: Any, y: Any, verbose=True) -> "XGBoostClassifier":
        import pandas as pd

        logger.info("Training XGBoost model")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = X

        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = y

        self.model.fit(X_train, y_train, verbose=verbose)
        self._is_fitted = True
        logger.info("XGBoost training completed")
        return self

    def predict(self, X: Any) -> Any:
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info("Predicting with XGBoost")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        predictions = self.model.predict(X_pred)
        logger.info(f"Predictions shape: {predictions.shape}")
        return predictions

    def predict_proba(self, X: Any) -> Any:
        """예측 확률 반환"""
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        return self.model.predict_proba(X_pred)

    def __repr__(self) -> str:
        return f"XGBoostClassifier(params={self.params})"


class RandomForestClassifier:
    """랜덤 포레스트 분류 모델"""

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier as RFClassifier

        self.params = kwargs
        # 기본 파라미터 설정
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
        default_params.update(self.params)
        self.params = default_params

        self.model = RFClassifier(**self.params)
        self._is_fitted = False
        logger.info(f"Initializing RandomForest with params: {self.params}")

    def fit(self, X: Any, y: Any) -> "RandomForestClassifier":
        import pandas as pd

        logger.info("Training RandomForest model")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = X

        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = y

        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("RandomForest training completed")
        return self

    def predict(self, X: Any) -> Any:
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info("Predicting with RandomForest")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        predictions = self.model.predict(X_pred)
        logger.info(f"Predictions shape: {predictions.shape}")
        return predictions

    def predict_proba(self, X: Any) -> Any:
        """예측 확률 반환"""
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X

        return self.model.predict_proba(X_pred)

    def __repr__(self) -> str:
        return f"RandomForestClassifier(params={self.params})"


class NeuralNetworkRegressor:
    """PyTorch 기반 다중 출력 회귀 모델

    PyTorch를 사용한 신경망 회귀 모델.
    6개의 성능 메트릭을 동시에 예측하는 multi-output regression.
    """

    def __init__(
        self,
        hidden_layers: list = None,
        activation: str = 'relu',
        output_dim: int = 6,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        early_stopping_patience: int = 10,
        device: str = None,
        **kwargs
    ):
        """
        Args:
            hidden_layers: 은닉층 뉴런 수 리스트 (예: [128, 64, 32])
            activation: 활성화 함수 ('relu', 'tanh', 'sigmoid')
            output_dim: 출력 차원 (예측할 타겟 개수, 기본값 6)
            learning_rate: 학습률
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            verbose: 학습 로그 상세도
            early_stopping_patience: Early stopping patience
            device: 'cuda', 'cpu', 또는 None (자동 선택)
        """
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.activation = activation
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.kwargs = kwargs

        # Device 설정
        if device is None:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            import torch
            self.device = torch.device(device)

        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        self._is_fitted = False
        self._target_scaler = None
        self._feature_scaler = None

        logger.info(f"Initializing NeuralNetworkRegressor with hidden_layers={self.hidden_layers}, output_dim={self.output_dim}, device={self.device}")

    def _build_model(self, input_dim: int):
        """PyTorch 신경망 모델 구축"""
        import torch
        import torch.nn as nn

        logger.info(f"Building neural network with input_dim={input_dim}")

        class RegressionNetwork(nn.Module):
            def __init__(self, input_dim, hidden_layers, output_dim, activation):
                super(RegressionNetwork, self).__init__()

                layers = []
                prev_dim = input_dim

                # 은닉층 구성 (Batch Normalization 추가)
                for i, hidden_dim in enumerate(hidden_layers):
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))  # Batch Normalization 추가

                    # 활성화 함수
                    if activation == 'relu':
                        layers.append(nn.ReLU())
                    elif activation == 'tanh':
                        layers.append(nn.Tanh())
                    elif activation == 'sigmoid':
                        layers.append(nn.Sigmoid())
                    elif activation == 'leaky_relu':
                        layers.append(nn.LeakyReLU(0.2))

                    # Dropout 비율을 레이어 깊이에 따라 조정
                    dropout_rate = 0.3 if i < len(hidden_layers) // 2 else 0.2
                    layers.append(nn.Dropout(dropout_rate))
                    prev_dim = hidden_dim

                # 출력층
                layers.append(nn.Linear(prev_dim, output_dim))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        model = RegressionNetwork(input_dim, self.hidden_layers, self.output_dim, self.activation)
        model = model.to(self.device)

        logger.info(f"Model architecture:\n{model}")

        return model

    def fit(self, X: Any, y: Any) -> "NeuralNetworkRegressor":
        """모델 학습"""
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        logger.info("Training NeuralNetworkRegressor")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = np.array(X)

        if isinstance(y, pd.DataFrame):
            y_train = y.values
        else:
            y_train = np.array(y)

        # y가 1차원이면 2차원으로 변환
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        logger.info(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")

        # 특징 스케일링
        self._feature_scaler = StandardScaler()
        X_train_scaled = self._feature_scaler.fit_transform(X_train)

        # 타겟 스케일링
        self._target_scaler = StandardScaler()
        y_train_scaled = self._target_scaler.fit_transform(y_train)

        # Train/Validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train_scaled,
            test_size=self.validation_split,
            random_state=42
        )

        # PyTorch 텐서로 변환
        X_tr_tensor = torch.FloatTensor(X_tr).to(self.device)
        y_tr_tensor = torch.FloatTensor(y_tr).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # DataLoader 생성
        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 모델 구축
        input_dim = X_train_scaled.shape[1]
        self.model = self._build_model(input_dim)

        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()

        # AdamW 옵티마이저 사용 (weight decay로 정규화 개선)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01  # L2 정규화
        )

        # 학습률 스케줄러 - Cosine Annealing + ReduceLROnPlateau 조합
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
        )

        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # 학습 모드
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 검증 모드
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Learning rate 조정
            scheduler.step(val_loss)

            # Early stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최적 모델 저장
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                # 최적 모델 복원
                self.model.load_state_dict(best_model_state)
                break

        self._is_fitted = True
        logger.info("NeuralNetworkRegressor training completed")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        return self

    def predict(self, X: Any) -> Any:
        """예측 수행"""
        import pandas as pd
        import numpy as np
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info("Predicting with NeuralNetworkRegressor")

        # DataFrame인 경우 numpy 배열로 변환
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = np.array(X)

        # 특징 스케일링
        X_pred_scaled = self._feature_scaler.transform(X_pred)

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X_pred_scaled).to(self.device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        # 스케일 복원
        predictions = self._target_scaler.inverse_transform(predictions_scaled)

        logger.info(f"Predictions shape: {predictions.shape}")

        return predictions

    def get_training_history(self):
        """학습 히스토리 반환"""
        if not self.history['train_loss']:
            logger.warning("No training history available")
            return None
        return self.history

    def __repr__(self) -> str:
        return f"NeuralNetworkRegressor(hidden_layers={self.hidden_layers}, output_dim={self.output_dim}, device={self.device})"

