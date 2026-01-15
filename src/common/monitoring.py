from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import setup_logging

logger = setup_logging()


class Monitor:
    def __init__(self) -> None:
        self.heartbeats: Dict[str, datetime] = {}
        self.alerts: List[Dict[str, Any]] = []

    def heartbeat(self, component_name: str) -> None:
        """
        Records a heartbeat for a component.
        """
        self.heartbeats[component_name] = datetime.now()
        logger.debug(f"Heartbeat received from {component_name}")

    def alert(
        self,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Logs an alert and stores it.
        """
        alert_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "metadata": metadata or {},
        }
        self.alerts.append(alert_entry)

        log_msg = f"ALERT [{level}]: {message} | Metadata: {metadata}"
        if level.upper() == "CRITICAL":
            logger.critical(log_msg)
        elif level.upper() == "ERROR":
            logger.error(log_msg)
        else:
            logger.warning(log_msg)

    def check_health(self, timeout_seconds: int = 60) -> bool:
        """
        Checks if all registered components have sent a heartbeat recently.
        """
        now = datetime.now()
        healthy = True
        for component, last_ts in self.heartbeats.items():
            if (now - last_ts).total_seconds() > timeout_seconds:
                self.alert(
                    "ERROR",
                    f"Component {component} heartbeat timeout",
                    {"last_seen": last_ts},
                )
                healthy = False
        return healthy


# Global monitor instance
monitor = Monitor()
