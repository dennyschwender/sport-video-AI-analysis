import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import requests
from pathlib import Path


class LLMResult(BaseModel):
    events: List[Dict[str, Any]]
    meta: Dict[str, Any]


class LLMBackendBase:
    """Abstract backend interface."""

    def analyze_commentary(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    def health_check(self) -> bool:
        """Check if backend is available."""
        return True


@dataclass
class SimulatedLLM(LLMBackendBase):
    """A tiny deterministic simulated backend for offline testing."""

    def analyze_commentary(self, text: str) -> Dict[str, Any]:
        # naive simulation: look for keywords and return events with timestamps
        events: List[Dict[str, Any]] = []
        lines = text.splitlines()
        t = 0
        for ln in lines:
            ln_lower = ln.lower()
            confidence = 0.9  # simulated confidence
            
            if "goal" in ln_lower or "score" in ln_lower:
                events.append({
                    "type": "goal", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            elif "assist" in ln_lower:
                events.append({
                    "type": "assist", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            elif "penalty" in ln_lower or "foul" in ln_lower:
                events.append({
                    "type": "penalty", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            elif "shot" in ln_lower or "shoots" in ln_lower:
                events.append({
                    "type": "shot", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            elif "save" in ln_lower or "saved" in ln_lower or "stops" in ln_lower:
                events.append({
                    "type": "save", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            elif "timeout" in ln_lower or "time out" in ln_lower:
                events.append({
                    "type": "timeout", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            elif "turnover" in ln_lower:
                events.append({
                    "type": "turnover", "timestamp": t, "description": ln.strip(),
                    "confidence": confidence
                })
            t += 5
        
        # simulate processing time and cost estimations
        start = time.time()
        time.sleep(0.05)
        processing_ms = int((time.time() - start) * 1000)
        cost_usd = 0.0
        
        return LLMResult(
            events=events,
            meta={
                "model": "simulated",
                "processing_ms": processing_ms,
                "cost_usd": cost_usd,
                "input_chars": len(text)
            }
        ).model_dump()


# OpenAI with function calling and JSON schema
try:
    from openai import OpenAI  # type: ignore
    
    @dataclass
    class OpenAIBackend(LLMBackendBase):
        api_key: str
        model: str = "gpt-4o-mini"
        max_retries: int = 3
        timeout: float = 60.0
        
        def __post_init__(self):
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        
        def health_check(self) -> bool:
            try:
                self.client.models.list()
                return True
            except Exception:
                return False
        
        def analyze_commentary(self, text: str) -> Dict[str, Any]:
            start = time.time()
            
            # JSON schema for structured output
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "event_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "timestamp": {"type": "number"},
                                        "description": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "team": {"type": "string"},
                                        "player": {"type": "string"}
                                    },
                                    "required": ["type", "timestamp", "description", "confidence"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["events"],
                        "additionalProperties": False
                    }
                }
            }
            
            system_prompt = """Extract sports events from the commentary text. Identify:
- Event type (goal, assist, shot, save, penalty, timeout, turnover, etc.)
- Timestamp in seconds
- Description
- Confidence score (0-1)
- Team name (if mentioned)
- Player name (if mentioned)

Return all detected events with accurate timestamps."""
            
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    response_format=response_format,
                    temperature=0
                )
                
                processing_ms = int((time.time() - start) * 1000)
                content = resp.choices[0].message.content
                parsed = json.loads(content)
                
                # Calculate cost (approximate)
                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                
                # Pricing for gpt-4o-mini (as of 2024): $0.150/1M input, $0.600/1M output
                cost_usd = (input_tokens * 0.150 / 1_000_000) + (output_tokens * 0.600 / 1_000_000)
                
                return LLMResult(
                    events=parsed.get("events", []),
                    meta={
                        "model": self.model,
                        "processing_ms": processing_ms,
                        "cost_usd": cost_usd,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "input_chars": len(text)
                    }
                ).model_dump()
                
            except Exception as e:
                return LLMResult(
                    events=[],
                    meta={
                        "model": self.model,
                        "error": str(e),
                        "processing_ms": int((time.time() - start) * 1000)
                    }
                ).model_dump()
except ImportError:
    pass


# Anthropic Claude with structured outputs
try:
    from anthropic import Anthropic  # type: ignore
    
    @dataclass
    class AnthropicBackend(LLMBackendBase):
        api_key: str
        model: str = "claude-3-5-sonnet-20241022"
        max_retries: int = 3
        timeout: float = 60.0
        
        def __post_init__(self):
            self.client = Anthropic(api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        
        def health_check(self) -> bool:
            try:
                # Anthropic doesn't have a simple health check, so we'll just check if client is valid
                return self.client is not None
            except Exception:
                return False
        
        def analyze_commentary(self, text: str) -> Dict[str, Any]:
            start = time.time()
            
            system_prompt = """Extract sports events from the commentary. For each event provide:
- type (goal, assist, shot, save, penalty, timeout, turnover, etc.)
- timestamp (seconds from start)
- description
- confidence (0-1)
- team (if mentioned)
- player (if mentioned)

Return JSON array of events."""
            
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
                
                processing_ms = int((time.time() - start) * 1000)
                content = resp.content[0].text
                
                # Try to parse JSON from response
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        events = parsed
                    elif isinstance(parsed, dict) and "events" in parsed:
                        events = parsed["events"]
                    else:
                        events = []
                except json.JSONDecodeError:
                    events = []
                
                # Calculate cost (Anthropic pricing for Claude 3.5 Sonnet)
                input_tokens = resp.usage.input_tokens if resp.usage else 0
                output_tokens = resp.usage.output_tokens if resp.usage else 0
                
                # Pricing: $3/MTok input, $15/MTok output
                cost_usd = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
                
                return LLMResult(
                    events=events,
                    meta={
                        "model": self.model,
                        "processing_ms": processing_ms,
                        "cost_usd": cost_usd,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "input_chars": len(text)
                    }
                ).model_dump()
                
            except Exception as e:
                return LLMResult(
                    events=[],
                    meta={
                        "model": self.model,
                        "error": str(e),
                        "processing_ms": int((time.time() - start) * 1000)
                    }
                ).model_dump()
except ImportError:
    pass


# Gemini Backend
try:
    import google.generativeai as genai
    
    @dataclass
    class GeminiBackend(LLMBackendBase):
        api_key: str
        model: str = "gemini-1.5-flash"
        max_retries: int = 3
        timeout: float = 60.0
        
        def __post_init__(self):
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        
        def health_check(self) -> bool:
            try:
                return self.client is not None
            except Exception:
                return False
        
        def analyze_commentary(self, text: str) -> Dict[str, Any]:
            start = time.time()
            
            prompt = f"""Extract sports events from the commentary. For each event provide:
- type (goal, assist, shot, save, penalty, timeout, turnover, etc.)
- timestamp (seconds from start)
- description
- confidence (0-1)
- team (if mentioned)
- player (if mentioned)

Return a JSON array of events.

Commentary:
{text}"""
            
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.3,
                        'max_output_tokens': 4096,
                    }
                )
                
                processing_ms = int((time.time() - start) * 1000)
                content = response.text
                
                # Try to parse JSON from response
                try:
                    # Extract JSON from markdown code blocks if present
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        events = parsed
                    elif isinstance(parsed, dict) and "events" in parsed:
                        events = parsed["events"]
                    else:
                        events = []
                except json.JSONDecodeError:
                    events = []
                
                # Calculate cost based on model
                # Gemini 1.5 Flash: $0.075/1M input, $0.30/1M output
                # Gemini 1.5 Pro: $1.25/1M input, $5.00/1M output  
                # Gemini 3 Pro Preview: $1.25/1M input, $5.00/1M output
                input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
                output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
                
                # Use Flash pricing by default (most common)
                model_lower = self.model.lower()
                if 'flash' in model_lower:
                    cost_usd = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
                else:  # Pro or Preview models
                    cost_usd = (input_tokens * 1.25 / 1_000_000) + (output_tokens * 5.00 / 1_000_000)
                
                return LLMResult(
                    events=events,
                    meta={
                        "model": self.model,
                        "processing_ms": processing_ms,
                        "cost_usd": cost_usd,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "input_chars": len(text)
                    }
                ).model_dump()
                
            except Exception as e:
                return LLMResult(
                    events=[],
                    meta={
                        "model": self.model,
                        "error": str(e),
                        "processing_ms": int((time.time() - start) * 1000)
                    }
                ).model_dump()

except ImportError:
    GeminiBackend = None


# Hugging Face Inference API
@dataclass
class HuggingFaceBackend(LLMBackendBase):
    api_key: Optional[str] = None
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_retries: int = 3
    timeout: float = 60.0
    
    def health_check(self) -> bool:
        try:
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = requests.get(url, headers=headers, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def analyze_commentary(self, text: str) -> Dict[str, Any]:
        start = time.time()
        
        prompt = f"""Extract sports events from this commentary. Return JSON array with fields: type, timestamp, description, confidence.

Commentary: {text}

JSON output:"""
        
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        try:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 1024, "return_full_text": False}
            }
            
            for attempt in range(self.max_retries):
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                
                if resp.status_code == 200:
                    break
                elif resp.status_code == 503:  # Model loading
                    time.sleep(2 ** attempt)
                    continue
                else:
                    resp.raise_for_status()
            
            processing_ms = int((time.time() - start) * 1000)
            result = resp.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = ""
            
            # Try to extract JSON
            try:
                events = json.loads(generated_text)
                if not isinstance(events, list):
                    events = []
            except json.JSONDecodeError:
                events = []
            
            # HF Inference API doesn't provide token counts, estimate cost
            input_tokens = len(text) // 4  # rough estimate
            output_tokens = len(generated_text) // 4
            cost_usd = (input_tokens * 0.0002 / 1000) + (output_tokens * 0.0006 / 1000)  # estimated
            
            return LLMResult(
                events=events,
                meta={
                    "model": self.model,
                    "processing_ms": processing_ms,
                    "cost_usd": cost_usd,
                    "input_chars": len(text)
                }
            ).model_dump()
            
        except Exception as e:
            return LLMResult(
                events=[],
                meta={
                    "model": self.model,
                    "error": str(e),
                    "processing_ms": int((time.time() - start) * 1000)
                }
            ).model_dump()


# Ollama (self-hosted)
@dataclass
class OllamaBackend(LLMBackendBase):
    endpoint: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: float = 60.0
    
    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def analyze_commentary(self, text: str) -> Dict[str, Any]:
        start = time.time()
        
        prompt = f"""Extract sports events from this commentary. Return a JSON array where each event has: type, timestamp (seconds), description, and confidence (0-1).

Commentary:
{text}

JSON output:"""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            resp = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            
            processing_ms = int((time.time() - start) * 1000)
            result = resp.json()
            response_text = result.get("response", "")
            
            # Parse JSON
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, list):
                    events = parsed
                elif isinstance(parsed, dict) and "events" in parsed:
                    events = parsed["events"]
                else:
                    events = []
            except json.JSONDecodeError:
                events = []
            
            return LLMResult(
                events=events,
                meta={
                    "model": self.model,
                    "processing_ms": processing_ms,
                    "cost_usd": 0.0,  # self-hosted, no per-request cost
                    "input_chars": len(text),
                    "eval_count": result.get("eval_count", 0)
                }
            ).model_dump()
            
        except Exception as e:
            return LLMResult(
                events=[],
                meta={
                    "model": self.model,
                    "error": str(e),
                    "processing_ms": int((time.time() - start) * 1000)
                }
            ).model_dump()
