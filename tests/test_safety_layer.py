import unittest
import time
from datetime import datetime
from src.execution.safety import SafetyLayer

class TestSafetyLayer(unittest.TestCase):
    def test_rate_limiting(self):
        safety = SafetyLayer(max_messages_per_second=2)
        
        # First two should pass
        res1, msg1 = safety.validate_weights({1: 0.1})
        self.assertTrue(res1)
        
        res2, msg2 = safety.validate_weights({1: 0.1})
        self.assertTrue(res2)
        
        # Third should fail
        res3, msg3 = safety.validate_weights({1: 0.1})
        self.assertFalse(res3)
        self.assertIn("rate limit", msg3)

    def test_kill_switch(self):
        safety = SafetyLayer(max_drawdown_limit=-0.01) # 1% limit
        
        # Initial equity
        safety.update_pnl(100.0)
        
        # Small drop - fine
        safety.update_pnl(99.5)
        res, msg = safety.validate_weights({1: 0.1})
        self.assertTrue(res)
        
        # Large drop - kill
        safety.update_pnl(98.0)
        self.assertTrue(safety.is_killed)
        
        res2, msg2 = safety.validate_weights({1: 0.1})
        self.assertFalse(res2)
        self.assertIn("Kill-switch", msg2)

if __name__ == "__main__":
    unittest.main()
