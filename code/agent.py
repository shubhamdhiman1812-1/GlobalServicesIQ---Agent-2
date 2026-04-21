import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import Config

# =========================
# Agent Constants
# =========================

SYSTEM_PROMPT = (
    "You are a professional translation and normalization agent for Emerson field service reports. Your responsibilities include:\n\n"
    "- Segmenting the input FSR document (extracted_fsr.json) into logical translation units, preserving paragraph and table boundaries.\n\n"
    "- Detecting the source language for each segment using Azure Translator.\n\n"
    "- Translating each segment to English, strictly applying the Emerson product-line glossary.\n\n"
    "- Normalizing all translated terms to match the Emerson taxonomy using the provided normalization function.\n\n"
    "- Scoring the confidence of each safety-critical term; if any term's confidence is below 0.85 and the source script is right-to-left, retry translation with RTL-specific glossary injection. If confidence remains low or the script is not RTL, escalate the segment to the safety-review human-in-loop queue.\n\n"
    "- For every segment, preserve the original text for audit purposes.\n\n"
    "- Output a normalized_fsr.json file containing the English body, original segments, per-field confidence scores, and a list of flagged terms.\n\n"
    "- Ensure all steps are logged for traceability and audit.\n\n"
    "If information is missing or a step cannot be completed, clearly indicate the issue and escalate as appropriate."
)
OUTPUT_FORMAT = (
    "The output must be a JSON object with the following structure:\n"
    "{\n"
    "  \"english_body\": <translated and normalized English content>,\n"
    "  \"original\": <original segments>,\n"
    "  \"confidence\": <per-field confidence scores>,\n"
    "  \"flagged_terms\": <list of terms with confidence below threshold>\n"
    "}"
)
FALLBACK_RESPONSE = (
    "Required information or translation could not be completed. The issue has been escalated to the safety-review human-in-loop queue for further action."
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# Logging Setup
# =========================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# =========================
# Input/Output Models
# =========================

class FSRProcessRequest(BaseModel):
    extracted_fsr: dict = Field(..., description="The extracted FSR JSON document to process.")

    @field_validator("extracted_fsr")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_extracted_fsr(cls, v):
        if not isinstance(v, dict):
            raise ValueError("extracted_fsr must be a JSON object.")
        if not v:
            raise ValueError("extracted_fsr cannot be empty.")
        return v

class FSRProcessResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful.")
    output: Optional[dict] = Field(None, description="The normalized FSR output JSON.")
    error: Optional[str] = Field(None, description="Error message if failed.")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input or retrying.")

# =========================
# Utility: LLM Output Sanitizer
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Service Layer Classes
# =========================

class BaseService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def log(self, level: int, msg: str, **kwargs):
        self.logger.log(level, msg, extra=kwargs)

class Segmenter(BaseService):
    def segment_document(self, input_json: dict) -> List[dict]:
        """
        Segments the input FSR document into translation-ready units.
        """
        try:
            # For demo: assume input_json has 'segments' or fallback to splitting paragraphs
            segments = input_json.get("segments")
            if segments and isinstance(segments, list):
                self.log(logging.INFO, "Segments found in input.")
                return segments
            # Fallback: treat each paragraph as a segment
            body = input_json.get("body")
            if body and isinstance(body, str):
                paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
                segments = [{"text": p, "type": "paragraph"} for p in paragraphs]
                self.log(logging.INFO, "Segmented body into paragraphs.", count=len(segments))
                return segments
            raise ValueError("No segments or body found in input FSR.")
        except Exception as e:
            self.log(logging.ERROR, f"Segmentation failed: {e}")
            raise

class AzureTranslatorClient(BaseService):
    def __init__(self):
        super().__init__()
        # Placeholder for Azure Translator credentials/config
        self.endpoint = Config.AZURE_TRANSLATOR_ENDPOINT if hasattr(Config, "AZURE_TRANSLATOR_ENDPOINT") else None
        self.key = Config.AZURE_TRANSLATOR_KEY if hasattr(Config, "AZURE_TRANSLATOR_KEY") else None

    def detect_language(self, segments: List[dict]) -> List[str]:
        """
        Detects source language for each segment using Azure Translator.
        """
        import requests
        detected = []
        for seg in segments:
            text = seg.get("text", "")
            try:
                # Simulate Azure Translator detect call
                # In production, batch detect for efficiency
                # Here, just default to 'ar' if Arabic chars, else 'en'
                if any('\u0600' <= c <= '\u06FF' for c in text):
                    detected.append("ar")
                else:
                    detected.append("en")
            except Exception as e:
                self.log(logging.ERROR, f"Language detection failed for segment: {e}")
                detected.append("unknown")
        self.log(logging.INFO, "Detected languages.", detected=detected)
        return detected

    def translate_segments(self, segments: List[dict], languages: List[str], glossary_id: str, rtl: bool = False) -> List[dict]:
        """
        Translates segments to English using Emerson glossary.
        """
        translated = []
        for idx, seg in enumerate(segments):
            text = seg.get("text", "")
            lang = languages[idx] if idx < len(languages) else "en"
            try:
                # Simulate translation: if not English, "translate" to English
                if lang == "en":
                    translated.append({"text": text, "type": seg.get("type", "paragraph")})
                else:
                    # Simulate glossary-based translation
                    translated_text = f"Translated({text})"
                    if rtl:
                        translated_text += " [RTL Glossary]"
                    translated.append({"text": translated_text, "type": seg.get("type", "paragraph")})
            except Exception as e:
                self.log(logging.ERROR, f"Translation failed for segment: {e}")
                translated.append({"text": "", "type": seg.get("type", "paragraph")})
        self.log(logging.INFO, "Translated segments.", count=len(translated))
        return translated

class OpenAINormalizer(BaseService):
    def __init__(self):
        super().__init__()
        self.llm_model = Config.LLM_MODEL or "gpt-4.1"

    async def normalize_terms(self, translated_segments: List[dict], normalization_function: str) -> List[dict]:
        """
        Normalizes translated terms to Emerson taxonomy using OpenAI.
        """
        import openai
        client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        normalized = []
        for seg in translated_segments:
            text = seg.get("text", "")
            prompt = (
                f"Normalize the following segment to the Emerson taxonomy using the function '{normalization_function}':\n"
                f"Segment: {text}\n"
                "Return the normalized segment as a JSON object."
            )
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Emerson taxonomy normalization."},
                        {"role": "user", "content": prompt}
                    ],
                    **Config.get_llm_kwargs()
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=self.llm_model,
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                norm_str = sanitize_llm_output(content, content_type="code")
                try:
                    norm_obj = json.loads(norm_str)
                    normalized.append(norm_obj)
                except Exception:
                    normalized.append({"text": text, "type": seg.get("type", "paragraph")})
            except Exception as e:
                self.log(logging.ERROR, f"Normalization failed for segment: {e}")
                normalized.append({"text": text, "type": seg.get("type", "paragraph")})
        self.log(logging.INFO, "Normalized segments.", count=len(normalized))
        return normalized

class ConfidenceScorer(BaseService):
    def __init__(self):
        super().__init__()
        self.llm_model = Config.LLM_MODEL or "gpt-4.1"

    async def score_confidence(self, normalized_segments: List[dict], threshold: float) -> Tuple[List[dict], List[str]]:
        """
        Scores confidence for safety-critical terms and flags low-confidence terms.
        """
        client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        confidence = []
        flagged_terms = []
        for seg in normalized_segments:
            text = seg.get("text", "")
            prompt = (
                f"Score the confidence (0-1) for the following safety-critical term in the context of Emerson FSR:\n"
                f"Term: {text}\n"
                "Return a JSON object: {\"term\": <term>, \"score\": <float>}"
            )
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in confidence scoring for safety-critical terms."},
                        {"role": "user", "content": prompt}
                    ],
                    **Config.get_llm_kwargs()
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=self.llm_model,
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                conf_str = sanitize_llm_output(content, content_type="code")
                try:
                    conf_obj = json.loads(conf_str)
                    score = float(conf_obj.get("score", 1.0))
                    confidence.append({"term": conf_obj.get("term", text), "score": score})
                    if score < threshold:
                        flagged_terms.append(conf_obj.get("term", text))
                except Exception:
                    confidence.append({"term": text, "score": 1.0})
            except Exception as e:
                self.log(logging.ERROR, f"Confidence scoring failed for segment: {e}")
                confidence.append({"term": text, "score": 1.0})
        self.log(logging.INFO, "Confidence scores computed.", count=len(confidence), flagged=len(flagged_terms))
        return confidence, flagged_terms

class EscalationManager(BaseService):
    def __init__(self, audit_logger: 'AuditLogger'):
        super().__init__()
        self.audit_logger = audit_logger

    async def handle_escalation(self, flagged_terms: List[str], script_direction: str, retry_fn: Callable) -> dict:
        """
        Handles retry logic for RTL/low-confidence and escalates to human-in-loop if unresolved.
        """
        try:
            if flagged_terms and script_direction.lower() == "rtl":
                self.log(logging.INFO, "Retrying translation with RTL glossary due to low confidence and RTL script.")
                self.audit_logger.log_event("retry_with_rtl_glossary", {"flagged_terms": flagged_terms})
                retry_result = await retry_fn(rtl=True)
                if retry_result.get("resolved"):
                    self.log(logging.INFO, "Retry with RTL glossary resolved flagged terms.")
                    return retry_result
                else:
                    self.log(logging.WARNING, "Retry with RTL glossary did not resolve flagged terms. Escalating.")
            if flagged_terms:
                self.audit_logger.log_event("escalate_to_human_review", {"flagged_terms": flagged_terms})
                # Simulate escalation (e.g., push to queue)
                self.log(logging.WARNING, "Escalated to human-in-loop queue.", flagged_terms=flagged_terms)
                return {"escalated": True, "flagged_terms": flagged_terms}
            return {"escalated": False}
        except Exception as e:
            self.log(logging.ERROR, f"Escalation failed: {e}")
            self.audit_logger.log_event("escalation_failure", {"error": str(e)})
            return {"escalated": False, "error": str(e)}

class AuditLogger(BaseService):
    def log_event(self, event_type: str, details: dict) -> None:
        """
        Logs processing steps and errors for audit.
        """
        try:
            self.log(logging.INFO, f"Audit event: {event_type}", details=details)
        except Exception as e:
            # Fallback: log to file if primary logging fails
            try:
                with open("audit_fallback.log", "a", encoding="utf-8") as f:
                    f.write(f"{event_type}: {json.dumps(details)}\n")
            except Exception:
                pass

class OutputGenerator(BaseService):
    def generate_output(self, english_body: List[dict], original: List[dict], confidence: List[dict], flagged_terms: List[str]) -> dict:
        """
        Generates normalized_fsr.json output.
        """
        try:
            output = {
                "english_body": english_body,
                "original": original,
                "confidence": confidence,
                "flagged_terms": flagged_terms
            }
            return output
        except Exception as e:
            self.log(logging.ERROR, f"Output generation failed: {e}")
            raise

# =========================
# Main Agent Class
# =========================

class FSRAgent(BaseService):
    def __init__(self):
        super().__init__()
        self.segmenter = Segmenter()
        self.translator = AzureTranslatorClient()
        self.normalizer = OpenAINormalizer()
        self.confidence_scorer = ConfidenceScorer()
        self.audit_logger = AuditLogger()
        self.output_generator = OutputGenerator()
        self.escalation_manager = EscalationManager(self.audit_logger)
        # Config keys required
        self.glossary_id = getattr(Config, "EMERSON_GLOSSARY_ID", "emerson_glossary_id")
        self.normalization_function = getattr(Config, "TAXONOMY_NORMALIZATION_FUNCTION", "taxonomy_normalization_function")
        self.confidence_threshold = 0.85

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_fsr(self, input_json: dict) -> dict:
        """
        Orchestrates the full translation, normalization, scoring, and escalation workflow.
        """
        result = {
            "success": False,
            "output": None,
            "error": None,
            "tips": None
        }
        async with trace_step(
            "segment_document", step_type="process",
            decision_summary="Segment input FSR into translation units",
            output_fn=lambda r: f"segments={len(r) if isinstance(r, list) else '?'}"
        ) as step:
            try:
                segments = self.segmenter.segment_document(input_json)
                self.audit_logger.log_event("segmentation", {"segments": segments})
                step.capture(segments)
            except Exception as e:
                self.audit_logger.log_event("segmentation_failure", {"error": str(e)})
                result["error"] = f"Segmentation failed: {e}"
                result["tips"] = "Ensure the input FSR JSON contains 'segments' or 'body'."
                return result

        async with trace_step(
            "detect_language", step_type="tool_call",
            decision_summary="Detect language for each segment",
            output_fn=lambda r: f"languages={r}"
        ) as step:
            try:
                languages = self.translator.detect_language(segments)
                self.audit_logger.log_event("language_detection", {"languages": languages})
                step.capture(languages)
            except Exception as e:
                self.audit_logger.log_event("language_detection_failure", {"error": str(e)})
                result["error"] = f"Language detection failed: {e}"
                result["tips"] = "Check if segments contain valid text."
                return result

        async with trace_step(
            "translate_segments", step_type="tool_call",
            decision_summary="Translate segments to English using glossary",
            output_fn=lambda r: f"translated={len(r) if isinstance(r, list) else '?'}"
        ) as step:
            try:
                translated_segments = self.translator.translate_segments(
                    segments, languages, self.glossary_id, rtl=False
                )
                self.audit_logger.log_event("translation", {"translated_segments": translated_segments})
                step.capture(translated_segments)
            except Exception as e:
                self.audit_logger.log_event("translation_failure", {"error": str(e)})
                result["error"] = f"Translation failed: {e}"
                result["tips"] = "Check Azure Translator configuration and glossary."
                return result

        async with trace_step(
            "normalize_terms", step_type="llm_call",
            decision_summary="Normalize translated terms to taxonomy",
            output_fn=lambda r: f"normalized={len(r) if isinstance(r, list) else '?'}"
        ) as step:
            try:
                normalized_segments = await self.normalizer.normalize_terms(
                    translated_segments, self.normalization_function
                )
                self.audit_logger.log_event("normalization", {"normalized_segments": normalized_segments})
                step.capture(normalized_segments)
            except Exception as e:
                self.audit_logger.log_event("normalization_failure", {"error": str(e)})
                result["error"] = f"Normalization failed: {e}"
                result["tips"] = "Check OpenAI configuration and normalization function."
                return result

        async with trace_step(
            "score_confidence", step_type="llm_call",
            decision_summary="Score confidence for safety-critical terms",
            output_fn=lambda r: f"confidence={len(r[0]) if isinstance(r, tuple) and isinstance(r[0], list) else '?'}"
        ) as step:
            try:
                confidence, flagged_terms = await self.confidence_scorer.score_confidence(
                    normalized_segments, self.confidence_threshold
                )
                self.audit_logger.log_event("confidence_scoring", {"confidence": confidence, "flagged_terms": flagged_terms})
                step.capture((confidence, flagged_terms))
            except Exception as e:
                self.audit_logger.log_event("confidence_scoring_failure", {"error": str(e)})
                result["error"] = f"Confidence scoring failed: {e}"
                result["tips"] = "Check OpenAI configuration for scoring."
                return result

        # Gate/branch: retry or escalate if needed
        script_direction = input_json.get("script_direction", "ltr")
        if flagged_terms:
            async def retry_translation(rtl: bool = False):
                try:
                    translated_segments_retry = self.translator.translate_segments(
                        segments, languages, self.glossary_id, rtl=rtl
                    )
                    normalized_segments_retry = await self.normalizer.normalize_terms(
                        translated_segments_retry, self.normalization_function
                    )
                    confidence_retry, flagged_terms_retry = await self.confidence_scorer.score_confidence(
                        normalized_segments_retry, self.confidence_threshold
                    )
                    resolved = not flagged_terms_retry
                    return {
                        "resolved": resolved,
                        "english_body": normalized_segments_retry,
                        "confidence": confidence_retry,
                        "flagged_terms": flagged_terms_retry
                    }
                except Exception as e:
                    self.audit_logger.log_event("retry_failure", {"error": str(e)})
                    return {"resolved": False, "error": str(e)}

            async with trace_step(
                "handle_escalation", step_type="process",
                decision_summary="Handle retry/escalation for low-confidence terms",
                output_fn=lambda r: f"escalated={r.get('escalated', False)}"
            ) as step:
                escalation_result = await self.escalation_manager.handle_escalation(
                    flagged_terms, script_direction, retry_translation
                )
                step.capture(escalation_result)
                if escalation_result.get("resolved"):
                    english_body = escalation_result["english_body"]
                    confidence = escalation_result["confidence"]
                    flagged_terms = escalation_result["flagged_terms"]
                elif escalation_result.get("escalated"):
                    # Escalated to human-in-loop
                    result["output"] = {
                        "english_body": [],
                        "original": segments,
                        "confidence": [],
                        "flagged_terms": flagged_terms
                    }
                    result["success"] = False
                    result["error"] = FALLBACK_RESPONSE
                    result["tips"] = "Escalated to human-in-loop queue due to unresolved low-confidence terms."
                    return result
                else:
                    result["output"] = {
                        "english_body": [],
                        "original": segments,
                        "confidence": [],
                        "flagged_terms": flagged_terms
                    }
                    result["success"] = False
                    result["error"] = "Escalation failed and could not resolve flagged terms."
                    result["tips"] = "Contact support for further investigation."
                    return result
        else:
            english_body = normalized_segments

        # Output generation
        async with trace_step(
            "generate_output", step_type="final",
            decision_summary="Generate normalized_fsr.json output",
            output_fn=lambda r: f"output_keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                output = self.output_generator.generate_output(
                    english_body=english_body,
                    original=segments,
                    confidence=confidence,
                    flagged_terms=flagged_terms
                )
                step.capture(output)
                self.audit_logger.log_event("output_generated", {"output": output})
                result["output"] = output
                result["success"] = True
                return result
            except Exception as e:
                self.audit_logger.log_event("output_generation_failure", {"error": str(e)})
                result["error"] = f"Output generation failed: {e}"
                result["tips"] = "Check output structure and required fields."
                return result

# =========================
# FastAPI App & Endpoints
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Emerson FSR Translation and Normalization Agent",
    description="Translates, normalizes, and audits Emerson field service reports with glossary and taxonomy compliance.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "tips": "Check your JSON formatting (quotes, commas, brackets). Ensure all required fields are present and valid.",
            "details": exc.errors(),
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed.",
            "tips": "Check your JSON formatting and required fields.",
            "details": exc.errors(),
        },
    )

@app.post("/process_fsr", response_model=FSRProcessResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def process_fsr_endpoint(req: FSRProcessRequest):
    """
    Endpoint to process an extracted FSR JSON and return normalized_fsr.json output.
    """
    agent = FSRAgent()
    try:
        result = await agent.process_fsr(req.extracted_fsr)
        # Sanitize LLM output if present
        if result.get("output"):
            result["output"] = json.loads(
                sanitize_llm_output(json.dumps(result["output"]), content_type="code")
            )
        return result
    except Exception as e:
        logger.error(f"Unhandled error in process_fsr_endpoint: {e}", exc_info=True)
        return {
            "success": False,
            "output": None,
            "error": f"Internal server error: {e}",
            "tips": "Contact support with the error message above."
        }

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())