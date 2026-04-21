
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

import httpx

from agent import app

@pytest.mark.asyncio
async def test_functional_successful_fsr_processing_via_process_fsr_endpoint():
    """
    Validates that the /process_fsr endpoint processes a well-formed FSR JSON and returns a successful, correctly structured response.
    """
    # Prepare input: valid extracted_fsr with 'segments'
    extracted_fsr = {
        "segments": [
            {"text": "Install valve in section A.", "type": "paragraph"},
            {"text": "تثبيت الصمام في القسم ب.", "type": "paragraph"}
        ],
        "script_direction": "ltr"
    }
    request_payload = {"extracted_fsr": extracted_fsr}

    # Patch OpenAI and trace_step context managers to avoid real LLM/network calls
    with patch("agent.OpenAINormalizer.normalize_terms", new_callable=AsyncMock) as mock_normalize_terms, \
         patch("agent.ConfidenceScorer.score_confidence", new_callable=AsyncMock) as mock_score_confidence, \
         patch("agent.trace_step") as mock_trace_step:

        # Mock trace_step as async context manager
        class FSRAgent:
            async def __aenter__(self): return self
            async def __aexit__(self, exc_type, exc, tb): return None
            def capture(self, _): pass
        mock_trace_step.return_value = FSRAgent()

        # Mock normalization: just return input segments as normalized
        mock_normalize_terms.return_value = [
            {"text": "Install valve in section A.", "type": "paragraph"},
            {"text": "FSRAgent(تثبيت الصمام في القسم ب.)", "type": "paragraph"}
        ]
        # Mock confidence scoring: all terms above threshold, no flagged terms
        mock_score_confidence.return_value = (
            [
                {"term": "Install valve in section A.", "score": 0.99},
                {"term": "FSRAgent(تثبيت الصمام في القسم ب.)", "score": 0.98}
            ],
            []
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/process_fsr", json=request_payload)
            assert resp.status_code == 200

            data = resp.json()
            assert data["success"] is True
            assert isinstance(data["output"], dict)
            for key in ("english_body", "original", "confidence", "flagged_terms"):
                assert key in data["output"]
            assert data["error"] is None