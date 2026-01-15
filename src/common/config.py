import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataConfig:
    base_path: str = "data"
    market_path: str = "data/market"
    event_path: str = "data/event"
    ism_db: str = "data/trading_system.db"
    forecast_db: str = "data/forecasts.db"
    target_weights_db: str = "data/target_weights.db"


@dataclass
class PortfolioConfig:
    risk_aversion: float = 1.0
    max_position: float = 0.2
    max_turnover: float = 1.0


@dataclass
class ExecutionConfig:
    simulated: bool = True
    transaction_cost_bps: float = 5.0
    max_weight: float = 0.2
    max_messages_per_second: int = 10
    max_drawdown_limit: float = -0.02
    max_adv_participation: float = 0.01


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def load(cls) -> "AppConfig":
        config_path = os.getenv("TRADING_SYSTEM_CONFIG", "config.yaml")
        raw_config: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f) or {}

        # Manual overrides from env vars (matching old Config logic)
        def _get_env_or_raw(section: str, key: str, default: Any) -> Any:
            env_key = f"TRADING_SYSTEM_{section.upper()}_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                return env_val
            return raw_config.get(section, {}).get(key, default)

        data = DataConfig(
            base_path=_get_env_or_raw("data", "base_path", "data"),
            market_path=_get_env_or_raw("data", "market_path", "data/market"),
            event_path=_get_env_or_raw("data", "event_path", "data/event"),
            ism_db=_get_env_or_raw("data", "ism_db", "data/trading_system.db"),
            forecast_db=_get_env_or_raw(
                "data", "forecast_db", "data/forecasts.db"
            ),
            target_weights_db=_get_env_or_raw(
                "data", "target_weights_db", "data/target_weights.db"
            ),
        )
        portfolio = PortfolioConfig(**raw_config.get("portfolio", {}))
        execution = ExecutionConfig(**raw_config.get("execution", {}))

        return cls(data=data, portfolio=portfolio, execution=execution)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Legacy support for string-based access.
        """
        keys = key.split(".")
        val: Any = self
        for k in keys:
            if hasattr(val, k):
                val = getattr(val, k)
            else:
                return default
        return val


# Global config instance
config = AppConfig.load()
