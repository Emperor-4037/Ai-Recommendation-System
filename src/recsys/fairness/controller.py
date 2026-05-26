import numpy as np
from typing import List, Dict, Any

import redis
from recsys.config import get_settings

class FairnessController:
    """
    Exposure-aware fairness controller.
    Tracks exposure parity, and applies exposure caps or stochastic exploration for under-exposed users.
    Uses Redis for distributed state tracking.
    """
    def __init__(self, exposure_cap: int = 100):
        self.exposure_cap = exposure_cap
        settings = get_settings()
        try:
            # We connect synchronously since we are in a potentially sync environment
            # In an async app, this could block briefly or we could use redis.asyncio
            self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Redis unavailable for Fairness Tracking: {e}")

        
    def record_exposures(self, candidate_ids: List[int]):
        if not candidate_ids:
            return
        pipeline = self.redis_client.pipeline()
        for cid in candidate_ids:
            pipeline.incr(f"exposure:{cid}")
        pipeline.execute()
            
    def get_exposure(self, candidate_id: int) -> int:
        val = self.redis_client.get(f"exposure:{candidate_id}")
        return int(val) if val else 0
        
    def apply_exposure_caps(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes candidates who have exceeded the exposure cap.
        """
        return [c for c in candidates if self.get_exposure(c['cand_id']) < self.exposure_cap]
        
    def adjust_scores_for_fairness(self, candidates: List[Dict[str, Any]], scores: np.ndarray) -> np.ndarray:
        """
        Boosts scores slightly for under-exposed candidates.
        """
        adjusted = scores.copy()
        for i, cand in enumerate(candidates):
            exp = self.get_exposure(cand['cand_id'])
            # Soft boost for under-exposed (e.g. fewer than 10 exposures)
            if exp < 10:
                adjusted[i] += 0.05
        return adjusted
