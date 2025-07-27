from __future__ import annotations
import os
import json
import base64
from io import BytesIO
from typing import Any, List, Optional

from dotenv import load_dotenv
from PIL import Image
from jinja2 import Environment
from pydantic import BaseModel, Field
from openai import OpenAI

class KeepOnly(BaseModel):
    keep: bool

class GPTInterfaceFC:

    def __init__(
        self,
        template_env: Environment,
        system_prompt_inside: str = "system_prompt_inside.txt",
        user_prompt: str = "user_prompt.txt",
        model: str = "gpt-4.1",
        temperature: float = 0.7,
    ):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature  = temperature
        self.template_env = template_env

        self.system_inside_tmpl = template_env.get_template(system_prompt_inside)
        self.user_tmpl = template_env.get_template(user_prompt)

    def _pil_to_b64(self, img: Image.Image) -> str:
        """Convert a PIL image to a base64 JPEG string suitable for data-URLs."""
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def _tool_spec(self) -> list[dict[str, Any]]:
        """Return the JSON schema that declares the verify_label function expects only 'keep'."""
        return [{
            "type": "function",
            "function": {
                "name": "verify_label",
                "description": "Return only the boolean 'keep' field.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keep": {
                            "type": "boolean",
                            "description": "True to keep this detection, false to discard it."
                        }
                    },
                    "required": ["keep"]
                },
            },
        }]

    def query_inside_fc(
        self,
        *,
        image: Image.Image,
        label: str,
        score: float,
        box: List[float],
        render_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[dict, dict]:
        
        render_kwargs = render_kwargs or {}
        base64_img    = self._pil_to_b64(image)

        system_prompt = self.system_inside_tmpl.render(**render_kwargs)
        user_prompt   = self.user_tmpl.render(**render_kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    },
                ],
            },
        ]

        resp = self.client.chat.completions.create(
            model       = self.model,
            temperature = self.temperature,
            messages    = messages,
            tools       = self._tool_spec(),
            tool_choice = {
                "type": "function",
                "function": {"name": "verify_label"},
            },
        )

        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise RuntimeError("Assistant did not call verify_label")

        tool_args = json.loads(calls[0].function.arguments)
        validated = KeepOnly.model_validate(tool_args)

        final = {
            "label": label,
            "score": score,
            "box":   box,
            "keep":  validated.keep
        }

        usage = resp.usage.model_dump()
        return final, usage