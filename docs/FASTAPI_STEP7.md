# FastAPI Integration for Step 7 Fashion Assistant

This package keeps the project frozen at Step 7 and exposes the LangGraph workflow through REST endpoints.

Use this workflow file:

```text
app/graph_workflow_step7.py
```

Do not use Step 8 for this version.

## Install API dependencies

```bash
pip install -r requirements_api.txt
```

You still need the original ML dependencies from your notebook environment, such as PyTorch, Transformers, FAISS, Pandas, LangGraph, and any model-specific packages already used by the project.

## Run the API

From the project root:

```bash
uvicorn app.api.main:app --reload
```

or:

```bash
python run_api.py
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### Health check

```text
GET /health
```

### Text-only search

```text
POST /search
```

Example JSON body:

```json
{
  "user_query": "black and red plaid dress with Peter Pan collar",
  "retrieval_top_k": 30,
  "display_top_k": 6,
  "rerank_top_k": 50,
  "debug": true
}
```

### Image or image+text search

```text
POST /search/image
```

Use form-data:

```text
user_query: find something similar
image: <uploaded image file>
retrieval_top_k: 30
display_top_k: 6
rerank_top_k: 50
debug: true
```

## What the API returns

The response includes:

```text
input_type
user_intent
retrieval_strategy
alpha
search_description
structured_query
quality_check
retry_count
retry_reason
products
conversation
debug_trace, only when debug=true
```

The API intentionally removes raw PIL image objects from the JSON response.
