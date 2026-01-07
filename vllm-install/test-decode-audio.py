#!/usr/bin/env python3
"""Test isolé du décodage audio et de la construction du prompt"""
import sys
import base64
import io
import numpy as np
from pydub import AudioSegment

# Copier les fonctions du serveur
AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

def decode_audio_base64(audio_b64: str) -> tuple[np.ndarray, int]:
    """Décode un audio base64 en numpy array"""
    if ',' in audio_b64:
        audio_b64 = audio_b64.split(',')[1]
    
    audio_bytes = base64.b64decode(audio_b64)
    
    # Essayer avec soundfile d'abord (WAV, FLAC, OGG)
    try:
        import soundfile as sf
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array, sample_rate
    except Exception as e:
        print(f"soundfile failed: {e}, trying pydub...")
    
    # Fallback pydub (MP3, M4A, AAC)
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    sample_rate = audio_segment.frame_rate
    audio_array = np.array(audio_segment.get_array_of_samples())
    
    # Normaliser
    if audio_segment.sample_width == 2:  # int16
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:  # int32
        audio_array = audio_array.astype(np.float32) / 2147483648.0
    
    # Stéréo -> Mono
    if audio_segment.channels == 2:
        audio_array = audio_array.reshape(-1, 2).mean(axis=1)
    
    return audio_array, sample_rate


def build_multimodal_prompt(messages: list[dict], audios: list, images: list) -> tuple[str, dict]:
    """Construit le prompt au format Qwen3-Omni"""
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    audio_idx = 0
    image_idx = 0
    mm_data = {}
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        prompt += f"<|im_start|>{role}\n"
        
        if isinstance(content, str):
            prompt += content
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                
                if item_type == "text":
                    prompt += item.get("text", "")
                
                elif item_type == "audio_url":
                    prompt += AUDIO_TOKEN + "\n"
                    audio_idx += 1
                
                elif item_type == "image_url":
                    prompt += "<|vision_start|><|image_pad|><|vision_end|>\n"
                    image_idx += 1
        
        prompt += "<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    
    if audios:
        mm_data["audio"] = audios
    if images:
        mm_data["image"] = images
    
    return prompt, mm_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test-decode-audio.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"=== Test de décodage audio ===")
    print(f"Fichier: {audio_file}\n")
    
    # Lire et encoder
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    audio_b64 = f"data:audio/m4a;base64,{base64.b64encode(audio_bytes).decode()}"
    
    print(f"✓ Audio encodé: {len(audio_b64)} caractères")
    
    # Décoder
    try:
        audio_array, sample_rate = decode_audio_base64(audio_b64)
        print(f"✓ Décodage réussi:")
        print(f"  - Shape: {audio_array.shape}")
        print(f"  - Dtype: {audio_array.dtype}")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Duration: {len(audio_array) / sample_rate:.2f}s")
        print(f"  - Min/Max: {audio_array.min():.3f} / {audio_array.max():.3f}")
    except Exception as e:
        print(f"✗ Erreur de décodage: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Construire le prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_b64}},
                {"type": "text", "text": "Décris ce que tu entends dans cet audio."}
            ]
        }
    ]
    
    audios = [(audio_array, sample_rate)]
    prompt, mm_data = build_multimodal_prompt(messages, audios, [])
    
    print(f"\n✓ Prompt construit:")
    print(f"  - Longueur: {len(prompt)} caractères")
    print(f"  - mm_data keys: {list(mm_data.keys())}")
    print(f"  - Audios dans mm_data: {len(mm_data.get('audio', []))}")
    
    print(f"\n✓ Prompt complet:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    # Vérifier que le token audio est présent
    if AUDIO_TOKEN in prompt:
        print(f"\n✓ Token audio trouvé dans le prompt!")
        print(f"  Position: {prompt.find(AUDIO_TOKEN)}")
    else:
        print(f"\n✗ ERREUR: Token audio NOT FOUND dans le prompt!")
        sys.exit(1)
    
    # Vérifier la correspondance
    token_count = prompt.count(AUDIO_TOKEN)
    audio_count = len(mm_data.get('audio', []))
    
    if token_count == audio_count:
        print(f"✓ Correspondance OK: {token_count} token(s) = {audio_count} audio(s)")
    else:
        print(f"✗ ERREUR: {token_count} token(s) != {audio_count} audio(s)")
        sys.exit(1)
    
    print(f"\n✅ Tous les tests passés! Format correct pour vLLM.")
