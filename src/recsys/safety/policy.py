from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SafetyGate:
    """
    Hard-filters candidates before final ranking.
    Removes blocked, high-report, unsafe content, and policy violations.
    """
    def __init__(self, max_safety_risk: float = 0.2):
        self.max_safety_risk = max_safety_risk
        
    def filter_candidates(self, user_id: int, candidate_profiles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns (safe_candidates, blocked_candidates)
        candidate_profiles must contain:
        - cand_id
        - is_blocked
        - report_count
        - safety_risk_score
        - intent
        """
        safe = []
        blocked = []
        
        user_intent = "casual" # In a real system, passed in or fetched
        
        for cand in candidate_profiles:
            reasons = []
            if cand.get('is_blocked', False):
                reasons.append("blocked")
            if cand.get('report_count', 0) > 3:
                reasons.append("high_report_count")
            if cand.get('safety_risk_score', 0.0) > self.max_safety_risk:
                reasons.append("high_safety_risk")
            if cand.get('has_unsafe_content', False):
                reasons.append("unsafe_content")
                
            # Incompatible intent pairings (e.g. strict long-term vs strict casual)
            cand_intent = cand.get('intent', 'unknown')
            if user_intent == "long_term" and cand_intent == "casual":
                reasons.append("incompatible_intent")
                
            if reasons:
                cand['_block_reasons'] = reasons
                blocked.append(cand)
            else:
                safe.append(cand)
                
        if blocked:
            logger.info(f"Filtered {len(blocked)} candidates for user {user_id} due to safety constraints.")
            
        return safe, blocked
