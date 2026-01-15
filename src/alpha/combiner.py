from typing import List, Dict


class SignalCombiner:
    @staticmethod
    def equal_weight(
        forecast_list: List[Dict[int, float]],
    ) -> Dict[int, float]:
        if not forecast_list:
            return {}

        combined = {}
        counts = {}

        for forecasts in forecast_list:
            for iid, val in forecasts.items():
                combined[iid] = combined.get(iid, 0.0) + val
                counts[iid] = counts.get(iid, 0) + 1

        return {
            iid: combined[iid] / len(forecast_list) for iid in combined.keys()
        }

    @staticmethod
    def weighted_average(
        forecast_list: List[Dict[int, float]], weights: List[float]
    ) -> Dict[int, float]:
        if not forecast_list or len(forecast_list) != len(weights):
            raise ValueError("Mismatched forecasts and weights")

        combined = {}
        for forecasts, weight in zip(forecast_list, weights):
            for iid, val in forecasts.items():
                combined[iid] = combined.get(iid, 0.0) + val * weight

        return combined
