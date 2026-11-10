import asyncio

import httpx


class LLMError(Exception):
    pass

class LLMClient:
    def __init__(self, client: httpx.AsyncClient, model: str, timeout: float, temperature: float):
        self.client = client
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    async def chat(self, system: str, user: str | list[dict], max_tokens: int, model : str | None = None) -> str:
        try:
            async with asyncio.timeout(self.timeout):
                r = await self.client.post("/chat/completions", json={
                    "model": model if model is not None else self.model,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                })
                r.raise_for_status()
                choice = r.json()["choices"][0]
                content = (choice["message"].get("content") or "").strip()
                if not content:
                    raise LLMError("Empty answer")
                finish_reason = choice["finish_reason"]
                if finish_reason == "length":
                    raise LLMError("Answer truncated")
                return content
        except LLMError:
            raise
        except TimeoutError:
            raise LLMError("Timeout error")
        except httpx.HTTPStatusError:
            raise LLMError(f"Failure status : {r.status_code}")
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error: {type(e).__name__}")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"Bad response")


