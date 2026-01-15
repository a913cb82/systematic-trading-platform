import os
from typing import Any, Dict

import yaml  # type: ignore
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = os.getenv("TRADING_SYSTEM_CONFIG", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value. Supports nested keys via dot notation.
        Order of precedence:
        1. Environment variable (e.g., TRADING_SYSTEM_DATA_PATH)
        2. Config file value
        3. Default value
        """
        # 1. Check environment variable first
        # (normalized to uppercase and underscores)
        env_key = f"TRADING_SYSTEM_{key.upper().replace('.', '_')}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val

        # 2. Check config dict
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


# Global config instance
config = Config()
