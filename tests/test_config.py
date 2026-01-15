import unittest
import os
from src.common.config import Config


class TestConfig(unittest.TestCase):
    def test_config_env_override(self):
        # Setup env var
        os.environ["TRADING_SYSTEM_DATA_MARKET_PATH"] = "env_path"

        # We need a new instance because it's a singleton-like in implementation
        # For testing, we just check the logic in Config().get
        cfg = Config()

        val = cfg.get("data.market_path")
        self.assertEqual(val, "env_path")

        # Test default
        self.assertEqual(cfg.get("non_existent", "def"), "def")


if __name__ == "__main__":
    unittest.main()
