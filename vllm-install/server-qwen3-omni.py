#!/usr/bin/env python3
"""
Serveur API vLLM pour Qwen3-Omni-30B
Compatible OpenAI API avec support audio/image
"""
import sys
import os

# Importer vLLM depuis le code source
# Paths are configured via environment variables set in vllm-serve.sh
_INSTALL_DIR = os.environ.get(
    'VLLM_INSTALL_DIR',
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, os.path.join(_INSTALL_DIR, 'vllm'))

# CRUCIAL : Forcer FlashInfer
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
from typing import Optional, List, Dict, Any

app = FastAPI(title="vLLM OpenAI-Compatible API")

# Configuration globale — path injected via VLLM_MODEL_PATH env var set in vllm-serve.sh
MODEL = os.environ.get('VLLM_MODEL_PATH', '/home/hci-ai/Documents/models/models/Qwen3-Omni-30B-A3B-Instruct/')
llm = None

@app.on_event("startup")
async def startup():
    """Charger le modèle au démarrage"""
    global llm
    print(f"\n{'='*50}")
    print(f"Chargement de {MODEL}...")
    print(f"Backend: FlashInfer")
    print(f"Ceci prendra ~6 minutes (60GB)...")
    print(f"{'='*50}\n")
    
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 2, "audio": 1}
    )
    
    print(f"\n{'='*50}")
    print(f"✅ Serveur prêt!")
    print(f"URL: http://0.0.0.0:8000")
    print(f"Health: http://0.0.0.0:8000/health")
    print(f"Completions: http://0.0.0.0:8000/v1/completions")
    print(f"Chat: http://0.0.0.0:8000/v1/chat/completions")
    print(f"{'='*50}\n")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL,
        "backend": "FlashInfer"
    }

@app.get("/v1/models")
async def list_models():
    """Liste des modèles disponibles (format OpenAI)"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": 1704672000,
                "owned_by": "vllm"
            }
        ]
    }

@app.post("/v1/completions")
async def completions(request: Request):
    """Endpoint completions (format OpenAI)"""
    global llm
    
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    
    # Format Qwen3-Omni avec stop sequences
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "Human:", "\nHuman:", "\n\nHuman"]
    )
    
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return {
        "id": "cmpl-" + os.urandom(12).hex(),
        "object": "text_completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": MODEL,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(generated_text.split()),
            "total_tokens": len(prompt.split()) + len(generated_text.split())
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Endpoint chat completions (format OpenAI) avec support streaming"""
    global llm
    
    data = await request.json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 2048)  # Augmenté de 150 à 2048 tokens
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    stream = data.get("stream", False)
    
    # Convertir messages au format Qwen3-Omni: <|im_start|>role\ncontent<|im_end|>
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if isinstance(content, str):
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif isinstance(content, list):
            # Support multimodal (texte + image/audio)
            text_content = ""
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get('text', '')
            if text_content:
                prompt += f"<|im_start|>{role}\n{text_content}<|im_end|>\n"
    
    # Ajouter le début de la réponse assistant
    prompt += "<|im_start|>assistant\n"
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    # Support du streaming
    if stream:
        async def generate_stream():
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # Simuler le streaming en envoyant par chunks (préserve les sauts de ligne)
            chunk_id = "chatcmpl-" + os.urandom(12).hex()
            
            # Découper en mots tout en préservant les sauts de ligne
            import re
            # Split sur espaces mais garde les \n
            tokens = re.split(r'(\s+)', generated_text)
            tokens = [t for t in tokens if t]  # Enlever les chaînes vides
            
            for i, token in enumerate(tokens):
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": MODEL,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant" if i == 0 else None,
                            "content": token
                        } if i == 0 else {
                            "content": token
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Dernier chunk avec finish_reason
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    # Mode non-streaming (réponse complète)
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(generated_text.split()),
            "total_tokens": len(prompt.split()) + len(generated_text.split())
        }
    }

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Démarrage du serveur vLLM")
    print(f"Modèle: {MODEL}")
    print(f"Backend: FlashInfer (VLLM_ATTENTION_BACKEND)")
    print(f"{'='*50}\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
