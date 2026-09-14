"""Amazon Bedrock coaching stub — ready when COACHING_PROVIDER = \"bedrock\".

Structurally correct boto3 bedrock-runtime invoke_model call. Not used until
config switches providers and real AWS credentials are available.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from app.config import BEDROCK_MODEL_ID, BEDROCK_REGION
from app.services.coaching.base import build_coaching_prompt


def get_coaching_text(deviation_data: dict[str, Any]) -> str:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or BEDROCK_REGION
    model_id = os.environ.get("BEDROCK_MODEL_ID", BEDROCK_MODEL_ID)
    prompt = build_coaching_prompt(deviation_data)

    # Claude Messages API body shape expected by Anthropic models on Bedrock.
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 120,
        "temperature": 0.4,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    payload = json.loads(response["body"].read())
    content = payload.get("content") or []
    if not content or "text" not in content[0]:
        raise RuntimeError("Bedrock returned an empty coaching response")

    return str(content[0]["text"]).strip()
