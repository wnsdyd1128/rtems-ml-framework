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

    def save(self, path: str):
        """모델 저장"""
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'output_dim': self.output_dim,
            'feature_scaler': self._feature_scaler,
            'target_scaler': self._target_scaler,
            'history': self.history,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'validation_split': self.validation_split,
                'early_stopping_patience': self.early_stopping_patience
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = None) -> "NeuralNetworkRegressor":
        """체크포인트에서 모델 로드"""
        import torch

        checkpoint = torch.load(path, map_location='cpu')

        # 모델 인스턴스 생성
        model = cls(
            hidden_layers=checkpoint['hidden_layers'],
            activation=checkpoint.get('activation', 'relu'),
            output_dim=checkpoint['output_dim'],
            device=device,
            **checkpoint.get('hyperparameters', {})
        )

        # 스케일러 및 히스토리 복원
        model._feature_scaler = checkpoint['feature_scaler']
        model._target_scaler = checkpoint['target_scaler']
        model.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})

        # 모델 구조 빌드 및 가중치 로드
        input_dim = model._feature_scaler.n_features_in_
        model.model = model._build_model(input_dim)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model._is_fitted = True

        logger.info(f"Model loaded from {path}")

        return model

    def __repr__(self) -> str:
        return f"NeuralNetworkRegressor(hidden_layers={self.hidden_layers}, output_dim={self.output_dim}, device={self.device})"


class VGGRegressor:
    """VGG-style Deep Neural Network Regressor

    VGG 논문의 핵심 아이디어를 1D 회귀 문제에 적용:
    - 작은 크기의 레이어를 깊게 쌓음 (VGG-11, VGG-16, VGG-19 스타일)
    - 모든 레이어에 Batch Normalization 적용
    - 블록 단위 구조 (각 블록은 동일한 크기의 레이어 반복)
    - Dropout과 Weight Decay를 통한 정규화

    Architecture presets:
    - VGG-11: [512, 512, 256, 256, 128, 128, 64, 64, 32]
    - VGG-16: [512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32]
    - Custom: 사용자 정의 구조
    """

    # VGG 스타일 아키텍처 프리셋
    ARCHITECTURES = {
        'VGG-11': [512, 512, 256, 256, 128, 128, 64, 64, 32],
        'VGG-16': [512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32],
        'VGG-19': [512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64, 64, 64, 32],
    }

    def __init__(
        self,
        architecture: str = 'VGG-11',
        hidden_layers: list = None,
        output_dim: int = 6,
        learning_rate: float = 0.001,
        epochs: int = 500,
        batch_size: int = 16,
        validation_split: float = 0.2,
        verbose: int = 1,
        early_stopping_patience: int = 50,
        dropout_rate: float = 0.3,
        weight_decay: float = 0.01,
        device: str = None,
        **kwargs
    ):
        """
        Args:
            architecture: VGG 아키텍처 프리셋 ('VGG-11', 'VGG-16', 'VGG-19')
            hidden_layers: 커스텀 레이어 구조 (None이면 architecture 사용)
            output_dim: 출력 차원 (예측할 타겟 개수)
            learning_rate: 초기 학습률
            epochs: 최대 학습 에포크 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            verbose: 학습 로그 상세도
            early_stopping_patience: Early stopping patience
            dropout_rate: Dropout 비율
            weight_decay: L2 정규화 계수
            device: 'cuda', 'cpu', 또는 None (자동 선택)
        """
        # 아키텍처 설정
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
            self.architecture = 'Custom'
        elif architecture in self.ARCHITECTURES:
            self.hidden_layers = self.ARCHITECTURES[architecture]
            self.architecture = architecture
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(self.ARCHITECTURES.keys())} or provide custom hidden_layers")

        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
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

        logger.info(f"Initializing VGGRegressor with architecture={self.architecture}, layers={self.hidden_layers}, output_dim={self.output_dim}, device={self.device}")

    def _build_model(self, input_dim: int):
        """VGG-style 신경망 모델 구축"""
        import torch
        import torch.nn as nn

        logger.info(f"Building VGG-style network with input_dim={input_dim}")

        class VGGRegressionNetwork(nn.Module):
            def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate):
                super(VGGRegressionNetwork, self).__init__()

                layers = []
                prev_dim = input_dim

                # VGG 스타일 블록 구성
                # 각 레이어마다 Linear -> BatchNorm -> ReLU -> Dropout
                for i, hidden_dim in enumerate(hidden_layers):
                    # Linear layer
                    layers.append(nn.Linear(prev_dim, hidden_dim))

                    # Batch Normalization (VGG에서는 원래 없었지만 현대적 개선)
                    layers.append(nn.BatchNorm1d(hidden_dim))

                    # ReLU activation (VGG의 핵심)
                    layers.append(nn.ReLU(inplace=True))

                    # Dropout for regularization
                    layers.append(nn.Dropout(dropout_rate))

                    prev_dim = hidden_dim

                # 출력층 (linear activation for regression)
                layers.append(nn.Linear(prev_dim, output_dim))

                self.network = nn.Sequential(*layers)

                # Weight initialization (VGG style - Xavier initialization)
                self._initialize_weights()

            def _initialize_weights(self):
                """Xavier initialization for weights"""
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            def forward(self, x):
                return self.network(x)

        model = VGGRegressionNetwork(input_dim, self.hidden_layers, self.output_dim, self.dropout_rate)
        model = model.to(self.device)

        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"VGG Model architecture:\n{model}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def fit(self, X: Any, y: Any) -> "VGGRegressor":
        """모델 학습"""
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        logger.info("Training VGGRegressor")

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

        # 손실 함수
        criterion = nn.MSELoss()

        # SGD with Momentum (VGG 논문의 원래 방식)
        # 하지만 더 나은 수렴을 위해 AdamW도 옵션으로 제공
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 학습률 스케줄러 - ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=self.verbose > 0,
            min_lr=1e-7
        )

        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training for {self.epochs} epochs...")

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
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                # 최적 모델 복원
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break

        self._is_fitted = True
        logger.info("VGGRegressor training completed")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        return self

    def predict(self, X: Any) -> Any:
        """예측 수행"""
        import pandas as pd
        import numpy as np
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info("Predicting with VGGRegressor")

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

    def save(self, path: str):
        """모델 저장"""
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'hidden_layers': self.hidden_layers,
            'output_dim': self.output_dim,
            'feature_scaler': self._feature_scaler,
            'target_scaler': self._target_scaler,
            'history': self.history,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate,
                'weight_decay': self.weight_decay
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = None) -> "VGGRegressor":
        """체크포인트에서 모델 로드"""
        import torch

        checkpoint = torch.load(path, map_location='cpu')

        # 모델 인스턴스 생성
        model = cls(
            architecture=checkpoint.get('architecture', 'Custom'),
            hidden_layers=checkpoint['hidden_layers'],
            output_dim=checkpoint['output_dim'],
            device=device,
            **checkpoint.get('hyperparameters', {})
        )

        # 스케일러 및 히스토리 복원
        model._feature_scaler = checkpoint['feature_scaler']
        model._target_scaler = checkpoint['target_scaler']
        model.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})

        # 모델 구조 빌드 및 가중치 로드
        input_dim = model._feature_scaler.n_features_in_
        model.model = model._build_model(input_dim)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model._is_fitted = True

        logger.info(f"Model loaded from {path}")

        return model

    def __repr__(self) -> str:
        return f"VGGRegressor(architecture={self.architecture}, layers={self.hidden_layers}, output_dim={self.output_dim}, device={self.device})"

