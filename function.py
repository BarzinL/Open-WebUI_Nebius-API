"""
title: Nebius AI Studio Pipe
authors: barzin
author_url: https://github.com/BarzinL
version: 0.1.0
required_open_webui_version: 0.5.10
license: AGPL-3.0-or-later
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator, Dict, AsyncIterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        NEBIUS_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "nebius"
        self.name = "nebius/"
        self.valves = self.Valves(**{"NEBIUS_API_KEY": os.getenv("NEBIUS_API_KEY", "")})
        self.base_url = "https://api.studio.nebius.ai/v1/"
        self.MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB per image
        self.SUPPORTED_IMAGE_TYPES = [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ]
        self.REQUEST_TIMEOUT = (3.05, 60)

    def get_nebius_models(self):
        return [
            {
                "id": "deepseek-ai/DeepSeek-R1",
                "name": "DeepSeek/R1",
                "supports_vision": False,
            },
            {
                "id": "Qwen/Qwen2.5-72B-Instruct",
                "name": "Qwen2.5 72B-Instruct",
                "supports_vision": False,
            },
            {
                "id": "Qwen/Qwen2-VL-72B-Instruct",
                "name": "Qwen2-VL 72B-Instruct",
                "supports_vision": True,
            },
            {
                "id": "aaditya/Llama3-OpenBioLLM-70B",
                "name": "Llama3 OpenBioLLM-70B",
                "supports_vision": False,
            },
            {
                "id": "meta-llama/Llama-3.3-70B-Instruct",
                "name": "Llama 3.3 70B-Instruct",
                "supports_vision": False,
            },
        ]

    def pipes(self) -> List[dict]:
        return self.get_nebius_models()

    def process_content(self, content: Union[str, List[dict]], model_id: str) -> str:
        if isinstance(content, str):
            return content

        processed_content = ""
        for item in content:
            if item["type"] == "text":
                processed_content += item["text"]
            elif item["type"] == "image_url" and "Qwen2-VL" in model_id:
                try:
                    base64_image = self.process_image(item)
                    processed_content += f"\n<image>{base64_image}</image>\n"
                except Exception as e:
                    print(f"Error processing image: {e}")
                    raise ValueError(f"Error processing image: {e}")
        return processed_content

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            if media_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported media type: {media_type}")
            image_size = len(base64_data) * 3 / 4
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds {self.MAX_IMAGE_SIZE/(1024*1024)}MB limit: {image_size/(1024*1024):.2f}MB"
                )
            return base64_data
        else:
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get("content-type", "")
            if content_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported media type: {content_type}")
            content_length = int(response.headers.get("content-length", 0))
            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image exceeds {self.MAX_IMAGE_SIZE/(1024*1024)}MB limit: {content_length/(1024*1024):.2f}MB"
                )
            img_response = requests.get(url)
            if img_response.status_code == 200:
                import base64

                return base64.b64encode(img_response.content).decode("utf-8")
            raise ValueError(f"Failed to fetch image from URL: {url}")

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.NEBIUS_API_KEY:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error: NEBIUS_API_KEY is required",
                            "done": True,
                        },
                    }
                )
            return "Error: NEBIUS_API_KEY is required"

        try:
            system_message, messages = pop_system_message(body["messages"])
            # Strip the prefix
            model_name = body["model"].replace("nebius_api.", "")
            processed_messages = []
            for message in messages:
                processed_content = self.process_content(
                    message.get("content", ""),
                    model_name,  # Use model_name instead of body["model"]
                )
                processed_messages.append(
                    {"role": message["role"], "content": processed_content}
                )
            if system_message:
                processed_messages.insert(
                    0, {"role": "system", "content": str(system_message)}
                )
            payload = {
                "model": model_name,  # Use model_name instead of body["model"]
                "messages": processed_messages,
                "temperature": body.get("temperature", 0.6),
                "max_tokens": body.get("max_tokens", 4096),
                "stream": body.get("stream", False),
            }

            headers = {
                "Authorization": f"Bearer {self.valves.NEBIUS_API_KEY}",
                "Content-Type": "application/json",
            }

            url = f"{self.base_url}chat/completions"

            if payload["stream"]:
                return self._stream_with_ui(
                    url, headers, payload, body, __event_emitter__
                )
            else:
                return self.non_stream_response(url, headers, payload)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg

    async def _stream_with_ui(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> Generator:
        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.REQUEST_TIMEOUT,
            ) as response:
                if response.status_code != 200:
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": "Request failed", "done": True},
                            }
                        )
                    yield f"Error: HTTP {response.status_code}: {response.text}"
                    return

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        if line.startswith(b"data: "):
                            line = line[6:]

                        data = json.loads(line)
                        if "choices" in data and len(data["choices"]) > 0:
                            chunk = None
                            if "delta" in data["choices"][0]:
                                chunk = data["choices"][0]["delta"].get("content", "")
                            elif "message" in data["choices"][0]:
                                chunk = data["choices"][0]["message"].get("content", "")
                            elif "text" in data["choices"][0]:
                                chunk = data["choices"][0]["text"]

                            if chunk:
                                # Filter out the [DONE] marker
                                chunk = chunk.replace("[DONE]", "")
                                if (
                                    chunk
                                ):  # Only yield if there's content after filtering
                                    yield chunk
                    except json.JSONDecodeError:
                        try:
                            line_str = line.decode("utf-8")
                            # Filter out the [DONE] marker from raw text
                            if line_str and not line_str.startswith("{"):
                                line_str = line_str.replace("[DONE]", "")
                                if (
                                    line_str
                                ):  # Only yield if there's content after filtering
                                    yield line_str
                        except Exception as e:
                            print(f"Debug - Error processing line: {str(e)}")
                            continue

                    time.sleep(0.01)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Request completed successfully",
                                "done": True,
                            },
                        }
                    )
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Stream error", "done": True},
                    }
                )
            print(f"Debug - Stream error: {str(e)}")  # Debug line
            yield f"Stream error: {str(e)}"

    def non_stream_response(self, url, headers, payload):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")
            res = response.json()
            return res["choices"][0]["message"]["content"] if "choices" in res else ""
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
