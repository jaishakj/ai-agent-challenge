#!/usr/bin/env python3
"""
LLM Client Wrapper Utility
Supports Google Gemini API and Groq API
"""

import os
import logging
from typing import Optional

# Google Gemini
from google.generativeai import configure, GenerativeModel

# Groq
from groq import Groq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMClient:
    def __init__(self, provider: str = "gemini", model: Optional[str] = None):
        """
        Initialize the LLM client wrapper.
        
        :param provider: "gemini" or "groq"
        :param model: optional model override
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "groq":
            self._init_groq()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_gemini(self):
        """Initialize Google Gemini API client"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

        configure(api_key=api_key)
        self.client = GenerativeModel(self.model or "gemini-1.5-flash")
        logger.info(f"Initialized Gemini with model {self.client.model_name}")

    def _init_groq(self):
        """Initialize Groq API client"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment variable")

        self.client = Groq(api_key=api_key)
        self.model = self.model or "mixtral-8x7b-32768"
        logger.info(f"Initialized Groq with model {self.model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from the configured provider.

        :param prompt: input text
        :param kwargs: extra params (like temperature, max tokens, etc.)
        :return: response text
        """
        if self.provider == "gemini":
            return self._generate_gemini(prompt, **kwargs)
        elif self.provider == "groq":
            return self._generate_groq(prompt, **kwargs)
        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

    def _generate_gemini(self, prompt: str, **kwargs) -> str:
        response = self.client.generate_content(prompt, **kwargs)
        return response.text.strip()

    def _generate_groq(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Example usage
    provider = os.getenv("LLM_PROVIDER", "gemini")  # default to gemini
    llm = LLMClient(provider=provider)

    result = llm.generate("Write a short poem about AI.")
    print(f"[{provider.upper()}] {result}")
