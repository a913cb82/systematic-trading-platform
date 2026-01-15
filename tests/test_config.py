import os
import unittest

from src.common.config import AppConfig


class TestConfig(unittest.TestCase):
    def test_config_env_override(self) -> None:
        # Setup env var
        os.environ["TRADING_SYSTEM_DATA_MARKET_PATH"] = "env_path"

        # We need a new instance because it's a singleton-like in
        # implementation
        # For testing, we just check the logic in AppConfig().get
        cfg = AppConfig.load()

        val = cfg.get("data.market_path")
        self.assertEqual(val, "env_path")

        # Test default
        self.assertEqual(cfg.get("non_existent", "def"), "def")


if __name__ == "__main__":
    unittest.main()
