
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import ValidationError

from agent import (
    FSRAgent,
    FSRProcessRequest,
    AzureTranslatorClient,
    OpenAINormalizer,
    OutputGenerator,
    FALLBACK_RESPONSE,
)

# --------------------------
# Fixtures
# --------------------------

@pytest.fixture
def sample_arabic_segment():
    return {'text': 'خطأ في النظام', 'type': 'paragraph'}

@pytest.fixture
def sample_segments(sample_arabic_segment):
    return [sample_arabic_segment]

@pytest.fixture
def fsr_input_rtl(sample_segments):
    return {
        'segments': sample_segments,
        'script_direction': 'rtl'
    }

@pytest.fixture
def fsr_input_no_segments_body():
    return {}

@pytest.fixture
def english_body():
    return [{'text': 'System error', 'type': 'paragraph'}]

@pytest.fixture
def original_body(sample_arabic_segment):
    return [sample_arabic_segment]

@pytest.fixture
def confidence_scores():
    return [{'term': 'System error', 'score': 0.92}]

# --------------------------
# Functional: Happy Path
# --------------------------

@pytest.mark.asyncio
async def test_fsragnt_process_fsr_happy_path(fsr_input_rtl):
    """
    Validates FSRAgent.process_fsr full workflow with well-formed input.
    All steps succeed, output is as expected.
    """
    agent = FSRAgent()

    # Patch translation, normalization, scoring to return deterministic values
    with patch.object(agent.segmenter, 'segment_document', return_value=fsr_input_rtl['segments']), \
         patch.object(agent.translator, 'detect_language', return_value=['ar']), \
         patch.object(agent.translator, 'translate_segments', return_value=[{'text': 'System error', 'type': 'paragraph'}]), \
         patch.object(agent.normalizer, 'normalize_terms', new=AsyncMock(return_value=[{'text': 'System error', 'type': 'paragraph'}])), \
         patch.object(agent.confidence_scorer, 'score_confidence', new=AsyncMock(return_value=([{'term': 'System error', 'score': 0.92}], []))), \
         patch.object(agent.output_generator, 'generate_output', wraps=agent.output_generator.generate_output):

        result = await agent.process_fsr(fsr_input_rtl)

    assert result['success'] is True
    assert result['output'] is not None
    for key in ['english_body', 'original', 'confidence', 'flagged_terms']:
        assert key in result['output']
    assert isinstance(result['output']['english_body'], list)
    assert isinstance(result['output']['original'], list)
    assert isinstance(result['output']['confidence'], list)
    assert isinstance(result['output']['flagged_terms'], list)
    assert result['error'] is None

# --------------------------
# Unit: Missing Segments/Body
# --------------------------

@pytest.mark.asyncio
async def test_fsragnt_process_fsr_missing_segments_and_body(fsr_input_no_segments_body):
    """
    Checks FSRAgent.process_fsr returns error if input lacks both 'segments' and 'body'.
    """
    agent = FSRAgent()
    # No patch needed: segmenter will raise ValueError
    result = await agent.process_fsr(fsr_input_no_segments_body)
    assert result['success'] is False
    assert result['output'] is None
    assert 'Segmentation failed' in result['error']
    assert 'segments' in result['tips'] or 'body' in result['tips']

# --------------------------
# Unit: AzureTranslatorClient.detect_language Arabic
# --------------------------

def test_azure_translator_client_detect_language_arabic(sample_segments):
    """
    Ensures AzureTranslatorClient.detect_language detects Arabic script.
    """
    client = AzureTranslatorClient()
    output = client.detect_language(sample_segments)
    assert output == ['ar']

# --------------------------
# Unit: OpenAINormalizer.normalize_terms handles LLM failure
# --------------------------

@pytest.mark.asyncio
async def test_openai_normalizer_normalize_terms_handles_llm_failure():
    """
    Verifies OpenAINormalizer.normalize_terms returns fallback if LLM fails or returns invalid JSON.
    """
    normalizer = OpenAINormalizer()
    segments = [{'text': 'Some text', 'type': 'paragraph'}]

    # Patch openai.AsyncAzureOpenAI to raise Exception
    with patch("agent.openai.AsyncAzureOpenAI", side_effect=Exception("LLM failure")):
        output = await normalizer.normalize_terms(segments, "taxonomy_normalization_function")
        assert isinstance(output, list)
        assert 'text' in output[0] and 'type' in output[0]
        assert output[0]['text'] == 'Some text'

    # Patch to return non-JSON output
    class FSRAgent:
        class FSRAgent:
            def __init__(self):
                self.message = type('msg', (), {'content': "not a json"})()
        choices = [FSRAgent()]
        usage = type('usage', (), {'prompt_tokens': 0, 'completion_tokens': 0})()
    dummy_client = MagicMock()
    dummy_client.chat.completions.create = AsyncMock(return_value=FSRAgent())
    with patch("agent.openai.AsyncAzureOpenAI", return_value=dummy_client):
        output = await normalizer.normalize_terms(segments, "taxonomy_normalization_function")
        assert isinstance(output, list)
        assert 'text' in output[0] and 'type' in output[0]
        assert output[0]['text'] == 'Some text'

# --------------------------
# Unit: FSRProcessRequest input validation
# --------------------------

def test_fsrprocessrequest_input_validation_blocks_non_dict():
    """
    Checks FSRProcessRequest raises ValidationError if 'extracted_fsr' is not a dict.
    """
    # Non-dict input: string
    with pytest.raises(ValidationError) as excinfo:
        FSRProcessRequest(extracted_fsr="not a dict")
    assert "extracted_fsr must be a JSON object" in str(excinfo.value)

    # Non-dict input: list
    with pytest.raises(ValidationError) as excinfo:
        FSRProcessRequest(extracted_fsr=[1,2,3])
    assert "extracted_fsr must be a JSON object" in str(excinfo.value)

# --------------------------
# Integration: Escalation on low confidence and RTL
# --------------------------

@pytest.mark.asyncio
async def test_fsragnt_process_fsr_escalates_on_low_confidence_and_rtl(fsr_input_rtl):
    """
    Tests FSRAgent.process_fsr triggers escalation when confidence < threshold and script_direction is 'rtl'.
    """
    agent = FSRAgent()
    flagged_terms = ['System error']
    # Patch all steps to succeed, but confidence scorer always returns flagged_terms
    with patch.object(agent.segmenter, 'segment_document', return_value=fsr_input_rtl['segments']), \
         patch.object(agent.translator, 'detect_language', return_value=['ar']), \
         patch.object(agent.translator, 'translate_segments', return_value=[{'text': 'System error', 'type': 'paragraph'}]), \
         patch.object(agent.normalizer, 'normalize_terms', new=AsyncMock(return_value=[{'text': 'System error', 'type': 'paragraph'}])), \
         patch.object(agent.confidence_scorer, 'score_confidence', new=AsyncMock(return_value=([{'term': 'System error', 'score': 0.81}], flagged_terms))), \
         patch.object(agent.output_generator, 'generate_output', wraps=agent.output_generator.generate_output):

        result = await agent.process_fsr(fsr_input_rtl)

    assert result['success'] is False
    assert result['output']['english_body'] == []
    assert result['output']['confidence'] == []
    assert result['output']['flagged_terms'] is not None and len(result['output']['flagged_terms']) > 0
    assert result['error'] == FALLBACK_RESPONSE

# --------------------------
# Unit: OutputGenerator.generate_output handles empty flagged_terms
# --------------------------

def test_output_generator_generate_output_handles_empty_flagged_terms(english_body, original_body, confidence_scores):
    """
    Ensures OutputGenerator.generate_output returns valid output dict when flagged_terms is empty.
    """
    generator = OutputGenerator()
    output = generator.generate_output(
        english_body=english_body,
        original=original_body,
        confidence=confidence_scores,
        flagged_terms=[]
    )
    assert isinstance(output, dict)
    assert isinstance(output['english_body'], list)
    assert isinstance(output['original'], list)
    assert isinstance(output['confidence'], list)
    assert output['flagged_terms'] == []

# --------------------------
# Integration: Translation failure
# --------------------------

@pytest.mark.asyncio
async def test_fsragnt_process_fsr_handles_translation_failure(fsr_input_rtl):
    """
    Checks FSRAgent.process_fsr returns error and tips if translation fails.
    """
    agent = FSRAgent()
    # Patch all steps up to translation to succeed, then translation fails
    with patch.object(agent.segmenter, 'segment_document', return_value=fsr_input_rtl['segments']), \
         patch.object(agent.translator, 'detect_language', return_value=['ar']), \
         patch.object(agent.translator, 'translate_segments', side_effect=Exception("Translation failed: Azure error")), \
         patch.object(agent.audit_logger, 'log_event'):

        result = await agent.process_fsr(fsr_input_rtl)

    assert result['success'] is False
    assert result['output'] is None
    assert 'Translation failed' in result['error']
    assert 'Azure Translator configuration' in result['tips']