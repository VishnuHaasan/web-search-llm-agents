import asyncio
import base64
import logging

import httpx

from fetch.safe_fetch import safe_get, IMAGE_CONTENT_TYPES, IMAGE_MAX_BYTES, FetchError
from llm.client import LLMClient, LLMError
from search.schema import ImageCandidate


async def describe_image(llm: LLMClient, client: httpx.AsyncClient, img: ImageCandidate, model: str, timeout: float) -> str | None:

    url = img.thumbnail or img.url

    try:
        async with asyncio.timeout(timeout):
            fetch_url, body, content_type = await safe_get(client, url, IMAGE_CONTENT_TYPES, IMAGE_MAX_BYTES)
            data = f"data:{content_type};base64,{base64.b64encode(body).decode()}"
            system = "Describe the image in one short sentence for a search result. Only describe what is visible."
            user = [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": data}},
            ]
            description = await llm.chat(system, user, max_tokens=100, model=model)
            return description
    except FetchError as e:
        logging.warning(e)
        return None
    except TimeoutError:
        logging.warning(f"Timeout while fetching image")
        return None
    except httpx.HTTPError as e:
        logging.warning(f"HTTP error while fetching image : {e!r}")
        return None
    except LLMError as e:
        logging.warning(f"LLM error while fetching image : {e!r}")
        return None

async def describe_images(llm: LLMClient, client: httpx.AsyncClient, images: list[ImageCandidate], model: str, limit: int, timeout: float) -> list[str | None]:

    to_collect = images[:limit]
    descriptions = await asyncio.gather(*[describe_image(llm, client, img, model, timeout) for img in to_collect])
    return list(descriptions) + [None] * max(0, len(images) - limit)





