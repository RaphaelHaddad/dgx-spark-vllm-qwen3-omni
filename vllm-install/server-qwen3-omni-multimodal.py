#!/usr/bin/env python3
"""
Serveur API OpenAI-compatible pour Qwen3-Omni-30B avec support multimodal (audio + image)
Basé sur vLLM + FastAPI + FlashInfer backend
"""

import sys
import os
import asyncio
import json
import base64
import io
from typing import AsyncIterator

# Ajouter vLLM au path
sys.path.insert(0, '/home/hci-ai/Documents/vllm-mvp-omni-spark/dgx-spark-vllm-qwen3-omni/vllm-install/vllm')

# CRITIQUE: Forcer FlashInfer avant import vLLM
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import numpy as np
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    
import soundfile as sf
from PIL import Image

# Configuration
MODEL = "/home/hci-ai/Documents/models/models/Qwen3-Omni-30B-A3B-Instruct/"
HOST = "0.0.0.0"
PORT = 8000

# Templates pour le multimodal
AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

app = FastAPI(title="Qwen3-Omni-30B Multimodal API")

# Variable globale pour le LLM
llm = None

@app.on_event("startup")
async def startup():
    """Initialisation du LLM au démarrage"""
    global llm
    
    print("🚀 Démarrage du serveur Qwen3-Omni-30B Multimodal...")
    print(f"📦 Modèle: {MODEL}")
    print(f"🎯 Backend: FlashInfer (forcé)")
    print(f"🔊 Support: Audio + Image + Text")
    
    # Charger le LLM avec support multimodal
    llm = LLM(
        model=MODEL,
        max_model_len=8192,
        max_num_seqs=5,
        gpu_memory_utilization=0.90,  # Réduit à 90% pour laisser de la marge
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 10, "image": 10},  # Max 10 audios/images par prompt
        enforce_eager=False,
    )
    
    print("✅ Modèle chargé avec succès!")
    print(f"📊 Contexte: 8192 tokens")
    print(f"🎤 Audio limit: 10 par prompt")
    print(f"🖼️  Image limit: 10 par prompt")


def decode_audio_base64(audio_b64: str) -> tuple[np.ndarray, int]:
    """
    Décode un audio base64 en array numpy + sample rate
    Supporte: WAV, MP3, FLAC, OGG, M4A, AAC
    """
    # Enlever le préfixe data:audio/...;base64, si présent
    if ',' in audio_b64:
        audio_b64 = audio_b64.split(',')[1]
    
    audio_bytes = base64.b64decode(audio_b64)
    
    # Essayer d'abord avec soundfile (WAV, FLAC, OGG)
    try:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        # Si soundfile échoue, utiliser pydub pour les autres formats (MP3, M4A, AAC)
        if not PYDUB_AVAILABLE:
            raise ValueError(f"Format audio non supporté par soundfile. Installez pydub: pip install pydub. Erreur: {e}")
        
        # Pydub peut lire M4A, MP3, AAC
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        sample_rate = audio.frame_rate
        
        # Convertir en numpy array
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normaliser de int16 à float32 [-1, 1]
        if audio.sample_width == 2:  # 16-bit
            audio_array = audio_array / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            audio_array = audio_array / 2147483648.0
        
        # Convertir stéréo en mono si nécessaire
        if audio.channels == 2:
            audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
    
    # Convertir en mono si stéréo (pour soundfile)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    return audio_array, sample_rate


def decode_image_base64(image_b64: str) -> Image.Image:
    """Décode une image base64 en PIL Image"""
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def build_multimodal_prompt(messages: list[dict], audios: list, images: list) -> tuple[str, dict]:
    """
    Construit le prompt au format Qwen3-Omni avec tokens multimodaux
    Retourne: (prompt_str, multi_modal_data)
    """
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    audio_idx = 0
    image_idx = 0
    mm_data = {}
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        prompt += f"<|im_start|>{role}\n"
        
        if isinstance(content, str):
            # Texte simple
            prompt += content
        elif isinstance(content, list):
            # Contenu multimodal
            for item in content:
                item_type = item.get("type")
                
                if item_type == "text":
                    prompt += item.get("text", "")
                
                elif item_type == "audio_url":
                    # Ajouter le token audio
                    prompt += AUDIO_TOKEN + "\n"
                    audio_idx += 1
                
                elif item_type == "image_url":
                    # Ajouter le token image
                    prompt += IMAGE_TOKEN + "\n"
                    image_idx += 1
        
        prompt += "<|im_end|>\n"
    
    # Ajouter le début de la réponse assistant
    prompt += "<|im_start|>assistant\n"
    
    # Préparer les données multimodales
    if audios:
        mm_data["audio"] = audios
    if images:
        mm_data["image"] = images
    
    return prompt, mm_data


@app.get("/health")
async def health():
    """Endpoint de santé"""
    return {
        "status": "healthy",
        "model": MODEL,
        "backend": "FlashInfer",
        "multimodal": ["audio", "image", "text"]
    }


@app.get("/v1/models")
async def list_models():
    """Liste les modèles disponibles"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": 1704067200,
                "owned_by": "Qwen",
                "capabilities": ["text", "audio", "image"]
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Endpoint chat completions avec support streaming et multimodal"""
    global llm
    
    data = await request.json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 2048)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    stream = data.get("stream", False)
    
    # Extraire les audios et images du contenu
    audios = []
    images = []
    
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                
                if item_type == "audio_url":
                    audio_url = item.get("audio_url", {}).get("url", "")
                    if audio_url.startswith("data:"):
                        # Décoder l'audio base64
                        audio_array, sample_rate = decode_audio_base64(audio_url)
                        audios.append((audio_array, sample_rate))
                
                elif item_type == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # Décoder l'image base64
                        pil_image = decode_image_base64(image_url)
                        images.append(pil_image)
    
    # Construire le prompt avec tokens multimodaux
    prompt, mm_data = build_multimodal_prompt(messages, audios, images)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    # Support du streaming
    if stream:
        async def generate_stream():
            inputs = {"prompt": prompt}
            if mm_data:
                inputs["multi_modal_data"] = mm_data
            
            outputs = llm.generate([inputs], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # Découper en tokens tout en préservant les sauts de ligne
            import re
            tokens = re.split(r'(\s+)', generated_text)
            tokens = [t for t in tokens if t]
            
            chunk_id = "chatcmpl-" + os.urandom(12).hex()
            
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
            
            # Dernier chunk
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
    
    # Mode non-streaming
    inputs = {"prompt": prompt}
    if mm_data:
        inputs["multi_modal_data"] = mm_data
    
    outputs = llm.generate([inputs], sampling_params)
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
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
