"""
Data Loader Strategies
다양한 데이터 소스를 로드하는 구체적인 전략 구현
"""

from typing import Any
from pathlib import Path
from loguru import logger

class CsvDataLoader:
    """CSV 파일 로더"""

    def load(self, source: str | Path) -> Any:
        logger.info(f"Loading CSV data from: {source}")
        # 실제 구현:
        # import pandas as pd
        # return pd.read_csv(source)
        return {"data": "csv_data", "source": str(source)}


class JsonDataLoader:
    """JSON 파일 로더"""

    def load(self, source: str | Path) -> Any:
        logger.info(f"Loading JSON data from: {source}")
        # 실제 구현:
        # import pandas as pd
        # return pd.read_json(source)
        return {"data": "json_data", "source": str(source)}


class RtemsJsonDataLoader:
    """RTEMS 스케줄링 실험 데이터 로더

    각 experiment는 가변 길이의 tasks 배열을 포함하며,
    각 task는 ID, CA(cache affinity), U(utilization) 정보를 가짐.
    label은 스케줄링 정책 (0: global, 1: clustered, 2: partitioned)
    """

    def load(self, source: str | Path) -> Any:
        """RTEMS JSON 데이터를 pandas DataFrame으로 로드

        Args:
            source: JSON 파일 경로

        Returns:
            pandas DataFrame with columns: experiment_id, tasks, label
        """
        import json
        import pandas as pd

        logger.info(f"Loading RTEMS experiment data from: {source}")

        with open(source, 'r') as f:
            data = json.load(f)

        # experiments 키가 있는지 확인
        if isinstance(data, dict) and 'experiments' in data:
            experiments = data['experiments']
        elif isinstance(data, list):
            experiments = data
        else:
            raise ValueError("JSON must contain 'experiments' key or be a list of experiments")

        logger.info(f"Loaded {len(experiments)} experiments")

        # DataFrame 생성
        df = pd.DataFrame(experiments)
        logger.debug(f"Raw data's df:\n {df.head()}")

        # 필수 컬럼 확인
        required_cols = ['tasks', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

        return df


class MemoryPatternRegressionDataLoader:
    """메모리 패턴 기반 성능 예측을 위한 회귀 데이터 로더

    각 샘플은 다음을 포함:
    - tasks: 가변 길이의 task 배열, 각 task는 memory_pattern과 ca를 포함
    - performance_metrics: 예측 타겟으로 사용될 성능 메트릭 (6개 실수값)
        - g.execution_time, g.turnaround_time
        - c.execution_time, c.turnaround_time
        - p.execution_time, p.turnaround_time
    """

    def load(self, source: str | Path) -> Any:
        """메모리 패턴 회귀 데이터를 pandas DataFrame으로 로드

        Args:
            source: JSON 파일 경로

        Returns:
            pandas DataFrame with columns: tasks, performance_metrics
        """
        import json
        import pandas as pd

        logger.info(f"Loading memory pattern regression data from: {source}")

        with open(source, 'r') as f:
            data = json.load(f)

        # 데이터 구조 확인
        if isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError("JSON must contain 'samples' key or be a list of samples")

        logger.info(f"Loaded {len(samples)} samples")

        # DataFrame 생성
        df = pd.DataFrame(samples)

        # 필수 컬럼 확인
        required_cols = ['tasks', 'performance_metrics']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 각 샘플의 task 수 통계
        task_counts = df['tasks'].apply(len)
        logger.info(f"Task count statistics: min={task_counts.min()}, max={task_counts.max()}, mean={task_counts.mean():.2f}")

        # performance_metrics 구조 확인
        sample_metrics = df['performance_metrics'].iloc[0]
        logger.info(f"Performance metrics keys: {list(sample_metrics.keys())}")

        logger.info(f"Data shape: {df.shape}")

        return df

