"""
Hyperparameter Tuning Strategies
하이퍼파라미터 최적화를 위한 구체적인 전략 구현
"""

from typing import Any, Dict
from loguru import logger


from .protocols import ModelProtocol



class GridSearchTuner:
    """그리드 서치 하이퍼파라미터 튜너"""
    
    def __init__(self, cv: int = 5, scoring: str = 'accuracy', n_jobs: int = -1):
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
    
    def tune(self, model: ModelProtocol, param_grid: Dict, X: Any, y: Any) -> ModelProtocol:
        logger.info(f"Running GridSearchCV with cv={self.cv}, scoring={self.scoring}")
        logger.info(f"Parameter grid: {param_grid}")
        
        # 실제 구현:
        # from sklearn.model_selection import GridSearchCV
        # grid_search = GridSearchCV(
        #     model, 
        #     param_grid, 
        #     cv=self.cv, 
        #     scoring=self.scoring,
        #     n_jobs=self.n_jobs
        # )
        # grid_search.fit(X, y)
        # return grid_search.best_estimator_
        
        # 예시: 최적 파라미터로 모델 재생성
        best_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
        logger.info(f"Best parameters found: {best_params}")
        
        model_class = type(model)
        return model_class(**best_params)
    
    def __repr__(self) -> str:
        return f"GridSearchTuner(cv={self.cv}, scoring={self.scoring}, n_jobs={self.n_jobs})"


class RandomSearchTuner:
    """랜덤 서치 하이퍼파라미터 튜너"""

    def __init__(self, n_iter: int = 10, cv: int = 5, scoring: str = 'accuracy', n_jobs: int = -1):
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def tune(self, model: ModelProtocol, param_grid: Dict, X: Any, y: Any) -> ModelProtocol:
        logger.info(f"Running RandomizedSearchCV with n_iter={self.n_iter}, cv={self.cv}")
        logger.info(f"Parameter distributions: {param_grid}")

        # 실제 구현:
        # from sklearn.model_selection import RandomizedSearchCV
        # random_search = RandomizedSearchCV(
        #     model,
        #     param_distributions=param_grid,
        #     n_iter=self.n_iter,
        #     cv=self.cv,
        #     scoring=self.scoring,
        #     n_jobs=self.n_jobs,
        #     random_state=42
        # )
        # random_search.fit(X, y)
        # return random_search.best_estimator_

        best_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
        logger.info(f"Best parameters found: {best_params}")

        model_class = type(model)
        return model_class(**best_params)

    def __repr__(self) -> str:
        return f"RandomSearchTuner(n_iter={self.n_iter}, cv={self.cv}, scoring={self.scoring})"


class BayesianOptimizationTuner:
    """베이지안 최적화 하이퍼파라미터 튜너"""

    def __init__(self, n_iter: int = 50, cv: int = 5, scoring: str = 'accuracy'):
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring

    def tune(self, model: ModelProtocol, param_grid: Dict, X: Any, y: Any) -> ModelProtocol:
        logger.info(f"Running Bayesian Optimization with n_iter={self.n_iter}")
        logger.info(f"Parameter space: {param_grid}")

        # 실제 구현:
        # from skopt import BayesSearchCV
        # bayes_search = BayesSearchCV(
        #     model,
        #     search_spaces=param_grid,
        #     n_iter=self.n_iter,
        #     cv=self.cv,
        #     scoring=self.scoring,
        #     random_state=42
        # )
        # bayes_search.fit(X, y)
        # return bayes_search.best_estimator_

        best_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
        logger.info(f"Best parameters found: {best_params}")

        model_class = type(model)
        return model_class(**best_params)

    def __repr__(self) -> str:
        return f"BayesianOptimizationTuner(n_iter={self.n_iter}, cv={self.cv}, scoring={self.scoring})"