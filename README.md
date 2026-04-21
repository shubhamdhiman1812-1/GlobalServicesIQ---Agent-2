# Emerson FSR Translation and Normalization Agent

A professional agent that translates, normalizes, and audits Emerson field service reports (FSRs). It segments input FSRs, detects language, translates using the Emerson glossary, normalizes terms to the Emerson taxonomy, scores confidence, and escalates low-confidence cases, returning a normalized JSON output with full auditability.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:
**Windows:**
```
.venv\Scripts\activate
```
**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values:
```
cp .env.example .env
```

### 5. Running the agent

- **Direct execution:**
  ```
  python code/agent.py
  ```
- **As a FastAPI server:**
  ```
  uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
  ```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`

**General**
- `ENVIRONMENT`

**Azure Key Vault (optional for production)**
- `USE_KEY_VAULT`
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_MODELS`
- `AZURE_OPENAI_ENDPOINT`

**API Keys / Secrets**
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_TRANSLATOR_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`
- `OBS_AZURE_SQL_PASSWORD`

**Service Endpoints**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_TRANSLATOR_ENDPOINT`
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`

**Observability DB**
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_PASSWORD`
- `OBS_AZURE_SQL_SCHEMA`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**Agent-Specific**
- `SERVICE_NAME`
- `SERVICE_VERSION`
- `VERSION`
- `VALIDATION_CONFIG_PATH`
- `EMERSON_GLOSSARY_ID`
- `TAXONOMY_NORMALIZATION_FUNCTION`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`

See `.env.example` for all required and optional variables.

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

### **POST** `/process_fsr`
- **Description:** Process an extracted FSR JSON and return normalized output.
- **Request body:**
  ```
  {
    "extracted_fsr": "object (required)"
  }
  ```
- **Response:**
  ```
  {
    "success": true|false,
    "output": {
      "english_body": [ ... ],
      "original": [ ... ],
      "confidence": [ ... ],
      "flagged_terms": [ ... ]
    } | null,
    "error": null|string,
    "tips": null|string
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t emerson-fsr-translation-agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name emerson-fsr-translation-agent emerson-fsr-translation-agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs emerson-fsr-translation-agent
```

### 7. Stop the container:
```
docker stop emerson-fsr-translation-agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Emerson FSR Translation and Normalization Agent** — Reliable, auditable translation and normalization for field service reports.
