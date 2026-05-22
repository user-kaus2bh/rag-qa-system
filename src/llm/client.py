import os
from typing import List, Iterator, Optional
from dataclasses import dataclass

from src.llm.prompts import Message, SYSTEM_PROMPT, format_chat_history


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if provider not in ("anthropic", "openai", "gemini"):
            raise ValueError(f"Unsupported provider: {provider}. Choose 'anthropic', 'openai', or 'gemini'.")

        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        if model:
            self.model = model
        elif provider == "anthropic":
            self.model = "claude-3-5-sonnet-20241022"
        elif provider == "openai":
            self.model = "gpt-4o-mini"
        else:
            self.model = "gemini-2.0-flash"

        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
                if not api_key:
                    raise EnvironmentError("ANTHROPIC_API_KEY not set in .env file.")
                self._client = anthropic.Anthropic(api_key=api_key)
            elif self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY", "")
                if not api_key:
                    raise EnvironmentError("OPENAI_API_KEY not set in .env file.")
                self._client = openai.OpenAI(api_key=api_key)
            else:
                from google import genai
                api_key = os.getenv("GEMINI_API_KEY", "")
                if not api_key:
                    raise EnvironmentError("GEMINI_API_KEY not set in .env file.")
                self._client = genai.Client(api_key=api_key)
        return self._client

    def complete(
        self,
        user_message: str,
        history: Optional[List[Message]] = None,
    ) -> LLMResponse:
        history_messages = [m if isinstance(m, Message) else Message(**m) for m in (history or [])]
        messages = format_chat_history(history_messages)
        messages.append({"role": "user", "content": user_message})

        if self.provider == "anthropic":
            return self._complete_anthropic(messages)
        elif self.provider == "openai":
            return self._complete_openai(messages)
        else:
            return self._complete_gemini(messages)

    def stream(
        self,
        user_message: str,
        history: Optional[List[Message]] = None,
    ) -> Iterator[str]:
        history_messages = [m if isinstance(m, Message) else Message(**m) for m in (history or [])]
        messages = format_chat_history(history_messages)
        messages.append({"role": "user", "content": user_message})

        if self.provider == "anthropic":
            yield from self._stream_anthropic(messages)
        elif self.provider == "openai":
            yield from self._stream_openai(messages)
        else:
            yield from self._stream_gemini(messages)

    def _build_gemini_contents(self, messages: List[dict]) -> List[dict]:
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents

    def _complete_gemini(self, messages: List[dict]) -> LLMResponse:
        from google.genai import types
        contents = self._build_gemini_contents(messages)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return LLMResponse(
            content=response.text,
            model=self.model,
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            stop_reason="stop",
        )

    def _stream_gemini(self, messages: List[dict]) -> Iterator[str]:
        from google.genai import types
        contents = self._build_gemini_contents(messages)
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        ):
            if chunk.text:
                yield chunk.text

    def _complete_anthropic(self, messages: List[dict]) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=messages,
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "",
        )

    def _complete_openai(self, messages: List[dict]) -> LLMResponse:
        system_msg = [{"role": "system", "content": self.system_prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=system_msg + messages,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            stop_reason=choice.finish_reason or "",
        )

    def _stream_anthropic(self, messages: List[dict]) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_openai(self, messages: List[dict]) -> Iterator[str]:
        system_msg = [{"role": "system", "content": self.system_prompt}]
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=system_msg + messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content