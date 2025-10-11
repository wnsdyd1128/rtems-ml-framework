"""
Preprocessing Strategies
데이터 전처리를 위한 구체적인 전략 구현
"""

from typing import Any, Tuple
from loguru import logger


class StandardScalerPreprocessor:
    """표준 스케일러 전처리기 (평균=0, 표준편차=1)"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, data: Any) -> "StandardScalerPreprocessor":
        import pandas as pd

        logger.info("Fitting StandardScaler")

        # DataFrame인 경우 피처 컬럼만 선택
        if isinstance(data, pd.DataFrame):
            # label, experiment_id 등 제외하고 숫자 피처만 선택
            exclude_cols = ['label', 'experiment_id']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            X = data[feature_cols]
        else:
            X = data

        self.scaler.fit(X)
        self._is_fitted = True
        logger.info(f"StandardScaler fitted on {X.shape[1]} features")
        return self

    def transform(self, data: Any) -> Any:
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        logger.info("Transforming data with StandardScaler")

        # DataFrame인 경우
        if isinstance(data, pd.DataFrame):
            exclude_cols = ['label', 'experiment_id']
            feature_cols = [col for col in data.columns if col not in exclude_cols]

            # 피처만 스케일링
            X_scaled = self.scaler.transform(data[feature_cols])

            # 결과를 DataFrame으로 만들고 원래 컬럼과 합침
            scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=data.index)

            # label, experiment_id 등 유지
            for col in exclude_cols:
                if col in data.columns:
                    scaled_df[col] = data[col].values

            # 원래 컬럼 순서 유지
            return scaled_df[data.columns]
        else:
            return self.scaler.transform(data)

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)


class MinMaxScalerPreprocessor:
    """Min-Max 스케일러 전처리기"""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self._is_fitted = False
    
    def fit(self, data: Any) -> "MinMaxScalerPreprocessor":
        logger.info(f"Fitting MinMaxScaler with range {self.feature_range}")
        # 실제 구현:
        # from sklearn.preprocessing import MinMaxScaler
        # self.scaler = MinMaxScaler(feature_range=self.feature_range)
        # self.scaler.fit(data)
        self._is_fitted = True
        return self
    
    def transform(self, data: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        logger.info("Transforming data with MinMaxScaler")
        # 실제 구현:
        # return self.scaler.transform(data)
        return {"transformed": data, "method": "minmax_scaler"}
    
    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)


class RobustScalerPreprocessor:
    """Robust 스케일러 전처리기 (이상치에 강건)"""

    def __init__(self):
        self._is_fitted = False

    def fit(self, data: Any) -> "RobustScalerPreprocessor":
        logger.info("Fitting RobustScaler")
        # 실제 구현:
        # from sklearn.preprocessing import RobustScaler
        # self.scaler = RobustScaler()
        # self.scaler.fit(data)
        self._is_fitted = True
        return self

    def transform(self, data: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        logger.info("Transforming data with RobustScaler")
        # 실제 구현:
        # return self.scaler.transform(data)
        return {"transformed": data, "method": "robust_scaler"}

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)


class RtemsFeatureEngineeringPreprocessor:
    """RTEMS 스케줄링 실험 데이터를 위한 피처 엔지니어링

    가변 길이 tasks 배열로부터 고정 길이 통계 피처를 생성:
    - CA (Cache Affinity) 통계: mean, std, min, max, median
    - U (Utilization) 통계: mean, std, min, max, median, sum
    - Task 개수
    """

    def __init__(self):
        self._is_fitted = False
        self._feature_names = None

    def fit(self, data: Any) -> "RtemsFeatureEngineeringPreprocessor":
        """피처 이름 학습 (통계적 변환은 학습 불필요)"""
        import pandas as pd

        logger.info("Fitting RtemsFeatureEngineeringPreprocessor")

        # 피처 이름 정의
        self._feature_names = [
            # 'task_count',
            'ca_mean', 'ca_std', 'ca_min', 'ca_max', 'ca_median',
            'u_mean', 'u_std', 'u_min', 'u_max', 'u_median', 'u_sum'
        ]

        self._is_fitted = True
        logger.info(f"Feature names: {self._feature_names}")
        return self

    def transform(self, data: Any) -> Any:
        """tasks 배열로부터 통계 피처 추출"""
        import pandas as pd
        import numpy as np

        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        logger.info("Transforming RTEMS data to statistical features")

        df = data.copy()

        # 각 experiment의 tasks로부터 통계 피처 추출
        features = []
        for idx, row in df.iterrows():
            tasks = row['tasks']

            # CA와 U 값 추출
            ca_values = [task['CA'] for task in tasks]
            u_values = [task['U'] for task in tasks]

            # 통계 계산
            feature_dict = {
                # 'task_count': len(tasks),
                'ca_mean': np.mean(ca_values),
                'ca_std': np.std(ca_values),
                'ca_min': np.min(ca_values),
                'ca_max': np.max(ca_values),
                'ca_median': np.median(ca_values),
                'u_mean': np.mean(u_values),
                'u_std': np.std(u_values),
                'u_min': np.min(u_values),
                'u_max': np.max(u_values),
                'u_median': np.median(u_values),
                'u_sum': np.sum(u_values)
            }
            features.append(feature_dict)

        # 피처 DataFrame 생성
        features_df = pd.DataFrame(features)

        # label 유지
        result_df = pd.concat([
            df[['label']].reset_index(drop=True),
            features_df
        ], axis=1)

        logger.info(f"Transformed data shape: {result_df.shape}")
        logger.info(f"Feature columns: {list(features_df.columns)}")

        return result_df

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)
    
