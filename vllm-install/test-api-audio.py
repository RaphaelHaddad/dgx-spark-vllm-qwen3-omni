#!/usr/bin/env python3
"""
Test de l'API vLLM avec analyse audio (Qwen3-Omni)
Nécessite un fichier audio en input
"""
import sys
import os
import requests
import json
import base64
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

def encode_audio_base64(audio_path: str) -> str:
    """Encode un fichier audio en base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_audio_analysis(audio_path: str):
    """Test l'analyse audio avec Qwen3-Omni"""
    print(f"=== Test API Audio - Qwen3-Omni ===")
    print(f"Fichier audio: {audio_path}")
    print(f"URL: {API_URL}\n")
    
    if not os.path.exists(audio_path):
        print(f"❌ Erreur: Fichier {audio_path} introuvable")
        return
    
    # Encoder l'audio en base64
    print("Encodage de l'audio en base64...")
    audio_base64 = encode_audio_base64(audio_path)
    print(f"Taille encodée: {len(audio_base64)} caractères\n")
    
    # Créer la requête multimodale
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyse cet audio et transcris son contenu. Décris ce que tu entends."
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{audio_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    print("Envoi de la requête à l'API...")
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Réponse de Qwen3-Omni ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("message", {}).get("content", "")
                print(f"\n=== Transcription/Analyse ===")
                print(text)
                print("\n✅ Test réussi!")
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test-api-audio.py <chemin_vers_fichier_audio.wav>")
        print("\nExemple:")
        print("  python test-api-audio.py test-audio.wav")
        print("  python test-api-audio.py ~/Music/sample.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    test_audio_analysis(audio_path)
