#!/usr/bin/env python3
"""
vLLM OpenAI-Compatible API Client Example
Demonstrates using vLLM's OpenAI-compatible API endpoints
"""

import requests
import json
from typing import Dict, List

class VLLMClient:
    """Simple client for vLLM OpenAI-compatible API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def list_models(self) -> List[Dict]:
        """List available models"""
        response = requests.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

    def complete(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict:
        """Generate completion"""

        # Get model name if not specified
        if model is None:
            models = self.list_models()
            model = models['data'][0]['id']

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=stream
        )
        response.raise_for_status()

        if stream:
            return response.iter_lines()
        else:
            return response.json()

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict:
        """Generate chat completion"""

        # Get model name if not specified
        if model is None:
            models = self.list_models()
            model = models['data'][0]['id']

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=stream
        )
        response.raise_for_status()

        if stream:
            return response.iter_lines()
        else:
            return response.json()


def main():
    # Initialize client
    client = VLLMClient("http://localhost:8000")

    print("="*60)
    print("vLLM API Client Examples")
    print("="*60)

    # Example 1: List models
    print("\n1. Listing available models...")
    models = client.list_models()
    for model in models['data']:
        print(f"   - {model['id']}")

    # Example 2: Simple completion
    print("\n2. Simple completion...")
    result = client.complete(
        prompt="The capital of France is",
        max_tokens=10,
        temperature=0.0
    )
    print(f"   Prompt: The capital of France is")
    print(f"   Response: {result['choices'][0]['text']}")

    # Example 3: Chat completion
    print("\n3. Chat completion...")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the Blackwell GPU architecture?"}
    ]
    result = client.chat(
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )
    print(f"   User: {messages[1]['content']}")
    print(f"   Assistant: {result['choices'][0]['message']['content']}")

    # Example 4: Streaming completion
    print("\n4. Streaming completion...")
    print("   Prompt: Write a short poem about AI")
    print("   Response: ", end="", flush=True)

    stream = client.complete(
        prompt="Write a short poem about AI",
        max_tokens=50,
        temperature=0.8,
        stream=True
    )

    for line in stream:
        if line:
            try:
                data = json.loads(line.decode('utf-8').removeprefix('data: '))
                if 'choices' in data and len(data['choices']) > 0:
                    token = data['choices'][0].get('text', '')
                    print(token, end="", flush=True)
            except (json.JSONDecodeError, AttributeError):
                pass

    print("\n")
    print("="*60)

if __name__ == "__main__":
    main()
