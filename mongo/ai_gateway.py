"""
Thin client for MongoDB's AI Gateway (ai.mongodb.com Model API Key), used by
the demo webapp to generate each simulated round's agent reasoning through
Atlas's own model endpoint instead of a separate LLM provider.

The exact request shape of ai.mongodb.com wasn't verified against a live
call (network probing from this dev environment was blocked), so this is
implemented as an OpenAI-compatible best-effort client: any failure (wrong
path, wrong auth scheme, unknown model) is swallowed and `reason()` returns
None so callers fall back to scripted reasoning instead of crashing the demo.
"""
from openai import OpenAI


class MongoAIGateway:
    def __init__(self, endpoint: str, api_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(base_url=endpoint.rstrip("/") + "/v1", api_key=api_key)

    def reason(self, system_prompt: str, user_prompt: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception:
            return None
