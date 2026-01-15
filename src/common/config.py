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
class AlpacaConfig:
    key: str = ""
    secret: str = ""
    paper: bool = True


@dataclass
class ExecutionConfig:
    provider: str = "mock"  # "mock" or "alpaca"
    simulated: bool = True
    transaction_cost_bps: float = 5.0
    max_weight: float = 0.2
    max_messages_per_second: int = 10
    max_drawdown_limit: float = -0.02
    max_adv_participation: float = 0.01
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)


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

        # Manual overrides from env vars
        def _get_env_or_raw(section: str, key: str, default: Any) -> Any:
            env_key = f"TRADING_SYSTEM_{section.upper()}_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                if isinstance(default, bool):
                    return env_val.lower() in ("true", "1", "yes")
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

        portfolio_raw = raw_config.get("portfolio", {})
        portfolio = PortfolioConfig(**portfolio_raw)

        execution_raw = raw_config.get("execution", {})
        alpaca_raw = execution_raw.get("alpaca", {})
        alpaca = AlpacaConfig(
            key=_get_env_or_raw(
                "execution", "alpaca_key", alpaca_raw.get("key", "")
            ),
            secret=_get_env_or_raw(
                "execution", "alpaca_secret", alpaca_raw.get("secret", "")
            ),
            paper=_get_env_or_raw(
                "execution", "alpaca_paper", alpaca_raw.get("paper", True)
            ),
        )

        execution = ExecutionConfig(
            provider=_get_env_or_raw(
                "execution", "provider", execution_raw.get("provider", "mock")
            ),
            simulated=_get_env_or_raw(
                "execution", "simulated", execution_raw.get("simulated", True)
            ),
            transaction_cost_bps=execution_raw.get(
                "transaction_cost_bps", 5.0
            ),
            max_weight=execution_raw.get("max_weight", 0.2),
            max_messages_per_second=execution_raw.get(
                "max_messages_per_second", 10
            ),
            max_drawdown_limit=execution_raw.get("max_drawdown_limit", -0.02),
            max_adv_participation=execution_raw.get(
                "max_adv_participation", 0.01
            ),
            alpaca=alpaca,
        )

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
