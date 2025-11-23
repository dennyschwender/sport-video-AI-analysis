from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class EventType(str, Enum):
    GOAL = "goal"
    ASSIST = "assist"
    SHOT = "shot"
    SAVE = "save"
    PENALTY = "penalty"
    FOUL = "foul"
    TURNOVER = "turnover"
    TIMEOUT = "timeout"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    SUBSTITUTION = "substitution"
    UNKNOWN = "unknown"


class Event(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    type: str
    timestamp: float = Field(..., alias='timestamp_seconds')
    description: str
    confidence: Optional[float] = None
    team: Optional[str] = None
    player: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    events: List[Dict[str, Any]]
    meta: Dict[str, Any]

    def parsed_events(self) -> List[Event]:
        """Normalize events into Event models with proper field mapping."""
        out = []
        for e in self.events:
            # support multiple possible timestamp keys
            ts = e.get('timestamp') or e.get('timestamp_seconds') or e.get('time') or 0
            normalized = {
                'type': e.get('type', 'unknown'),
                'timestamp_seconds': ts,
                'description': e.get('description', ''),
                'confidence': e.get('confidence'),
                'team': e.get('team'),
                'player': e.get('player'),
                'metadata': e.get('metadata')
            }
            try:
                out.append(Event(**normalized))
            except Exception:
                # fallback for malformed events
                out.append(Event(
                    type='unknown',
                    timestamp_seconds=ts,
                    description=str(e)
                ))
        return out

    def deduplicate_events(self, time_window: float = 5.0) -> List[Event]:
        """Remove duplicate events within a time window."""
        events = self.parsed_events()
        if not events:
            return []
        
        events.sort(key=lambda e: e.timestamp)
        deduplicated = [events[0]]
        
        for event in events[1:]:
            last = deduplicated[-1]
            # Consider events duplicates if same type within time window
            if event.type == last.type and abs(event.timestamp - last.timestamp) < time_window:
                # Keep the one with higher confidence or later timestamp
                if event.confidence and last.confidence:
                    if event.confidence > last.confidence:
                        deduplicated[-1] = event
                continue
            deduplicated.append(event)
        
        return deduplicated
