from datetime import datetime
from typing import Any, Dict, Optional

from .ism import InternalSecurityMaster


class SymbologyService:
    def __init__(self, ism: InternalSecurityMaster):
        self.ism = ism

    def map_to_internal_id(
        self,
        id_type: str,
        id_value: str,
        date: datetime,
        exchange: Optional[str] = None,
    ) -> Optional[int]:
        """
        Maps an external identifier to an internal_id.
        id_type can be 'TICKER', 'FIGI', 'ISIN', 'CUSIP', etc.
        If id_type is 'TICKER', exchange must be provided.
        """
        if id_type.upper() == "TICKER":
            if not exchange:
                raise ValueError("Exchange is required for TICKER lookup")
            return self.ism.get_internal_id(id_value, exchange, date)
        else:
            return self.ism.get_internal_id_by_external(
                id_type.upper(), id_value, date
            )

    def get_all_identifiers(
        self, internal_id: int, date: datetime
    ) -> Dict[str, Any]:
        """
        Returns all known identifiers for a given internal_id and date.
        """
        info = self.ism.get_symbol_info(internal_id, date)
        if not info:
            return {}

        identifiers = {"TICKER": info["ticker"], "EXCHANGE": info["exchange"]}

        # Get other identifiers from external_mappings
        ext_mappings = self.ism.get_external_mappings(internal_id, date)
        identifiers.update(ext_mappings)

        return identifiers
