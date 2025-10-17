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


class MemoryPatternPaddingPreprocessor:
    """메모리 패턴 데이터를 위한 패딩 전처리기

    가변 길이의 task 배열을 고정 길이로 변환:
    - memory_pattern의 값들만 추출 (메모리 주소는 무시)
    - ca (cache affinity) 값 추출
    - 패딩 시 실제 0과 null을 구분하기 위해 -999.0을 패딩 값으로 사용
    - 각 task당 최대 memory_pattern 개수를 찾아 패딩
    """

    def __init__(self, max_tasks: int = None, max_memory_entries: int = None, padding_value: float = -999.0):
        """
        Args:
            max_tasks: 최대 task 개수 (None이면 학습 데이터에서 자동 결정)
            max_memory_entries: 각 task당 최대 memory pattern 엔트리 수 (None이면 자동 결정)
            padding_value: 패딩에 사용할 값 (실제 0과 구분하기 위해 -999.0 사용)
        """
        self.max_tasks = max_tasks
        self.max_memory_entries = max_memory_entries
        self.padding_value = padding_value
        self._is_fitted = False
        self._target_columns = None

    def fit(self, data: Any) -> "MemoryPatternPaddingPreprocessor":
        """데이터로부터 최대 길이 학습"""
        import pandas as pd

        logger.info("Fitting MemoryPatternPaddingPreprocessor")

        df = data.copy()

        # 최대 task 수 결정
        if self.max_tasks is None:
            task_counts = df['tasks'].apply(len)
            self.max_tasks = int(task_counts.max())
            logger.info(f"Auto-detected max_tasks: {self.max_tasks}")

        # 최대 memory_pattern 엔트리 수 결정
        if self.max_memory_entries is None:
            max_entries = 0
            for _, row in df.iterrows():
                for task in row['tasks']:
                    if 'memory_pattern' in task:
                        max_entries = max(max_entries, len(task['memory_pattern']))
            self.max_memory_entries = max_entries
            logger.info(f"Auto-detected max_memory_entries: {self.max_memory_entries}")

        # 타겟 컬럼 이름 저장 (performance_metrics 구조 파악)
        if 'performance_metrics' in df.columns:
            sample_metrics = df['performance_metrics'].iloc[0]
            self._target_columns = []
            for key in ['g', 'c', 'p']:
                if key in sample_metrics:
                    for metric in ['execution_time', 'turnaround_time']:
                        if metric in sample_metrics[key]:
                            self._target_columns.append(f"{key}_{metric}")
            logger.info(f"Target columns: {self._target_columns}")

        self._is_fitted = True
        logger.info(f"Padding config: max_tasks={self.max_tasks}, max_memory_entries={self.max_memory_entries}, padding_value={self.padding_value}")
        return self

    def transform(self, data: Any) -> Any:
        """가변 길이 데이터를 고정 길이로 패딩"""
        import pandas as pd
        import numpy as np

        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        logger.info("Transforming data with padding")

        df = data.copy()

        # 각 샘플을 고정 길이 벡터로 변환
        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            tasks = row['tasks']

            # 각 task를 벡터로 변환
            task_vectors = []
            for task in tasks:
                # memory_pattern의 값들만 추출 (주소는 무시)
                memory_values = []
                if 'memory_pattern' in task and task['memory_pattern']:
                    memory_values = list(task['memory_pattern'].values())

                # 고정 길이로 패딩
                padded_memory = memory_values + [self.padding_value] * (self.max_memory_entries - len(memory_values))
                padded_memory = padded_memory[:self.max_memory_entries]  # 초과 시 자르기

                # ca 값 추가
                ca_value = task.get('ca', self.padding_value)

                # task 벡터: [memory_pattern_values..., ca]
                task_vector = padded_memory + [ca_value]
                task_vectors.append(task_vector)

            # task 수를 max_tasks로 패딩
            feature_dim = self.max_memory_entries + 1  # memory_entries + ca
            while len(task_vectors) < self.max_tasks:
                task_vectors.append([self.padding_value] * feature_dim)

            # 초과 시 자르기
            task_vectors = task_vectors[:self.max_tasks]

            # Flatten: [task0_mem0, task0_mem1, ..., task0_ca, task1_mem0, ...]
            X_vector = np.array(task_vectors).flatten()
            X_list.append(X_vector)

            # 타겟 값 추출 (있는 경우)
            if 'performance_metrics' in df.columns:
                metrics = row['performance_metrics']
                y_vector = []
                for col in self._target_columns:
                    key, metric = col.split('_', 1)
                    y_vector.append(metrics[key][metric])
                y_list.append(y_vector)

        # numpy 배열로 변환
        X = np.array(X_list)
        logger.info(f"Transformed X shape: {X.shape}")

        if y_list:
            y = np.array(y_list)
            logger.info(f"Transformed y shape: {y.shape}")
            logger.info(f"Target columns: {self._target_columns}")

            # X, y를 DataFrame으로 반환
            result_df = pd.DataFrame(X)
            for i, col in enumerate(self._target_columns):
                result_df[f'target_{col}'] = y[:, i]

            return result_df
        else:
            # 예측 시에는 X만 반환
            return pd.DataFrame(X)

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)

