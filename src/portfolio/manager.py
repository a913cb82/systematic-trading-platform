from datetime import datetime
from typing import Dict

from ..alpha.publisher import ForecastPublisher
from .optimizer import PortfolioOptimizer
from .publisher import TargetWeightPublisher


class PortfolioManager:
    def __init__(
        self,
        forecast_publisher: ForecastPublisher,
        optimizer: PortfolioOptimizer,
        weight_publisher: TargetWeightPublisher,
    ):
        self.forecast_publisher = forecast_publisher
        self.optimizer = optimizer
        self.weight_publisher = weight_publisher
        self.current_weights: Dict[int, float] = {}

        # Subscribe to forecasts
        self.forecast_publisher.subscribe_forecasts(self.on_forecast)

    def on_forecast(
        self, timestamp: datetime, forecasts: Dict[int, float]
    ) -> None:
        """
        Callback triggered when new forecasts are available.
        Runs optimization and publishes target weights.
        """
        target_weights = self.optimizer.optimize(
            timestamp, forecasts, self.current_weights
        )
        self.weight_publisher.submit_target_weights(timestamp, target_weights)

        # For PoC, assume target weights are achieved immediately.
        # In production, this would be updated via execution feedback.
        self.current_weights = target_weights
