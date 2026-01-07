#!/usr/bin/env python3
"""Test simple de Qwen3-Omni avec vLLM sur GB10"""

import sys
import os
sys.path.insert(0, '/home/oho/Projects/dgx-spark-vllm-setup/vllm-install/vllm')

# Force FlashInfer backend (FA2 PTX incompatible with sm_121)
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from vllm import LLM, SamplingParams

# Initialisation du modèle
print("Chargement du modèle Qwen3-Omni-30B-A3B-Instruct avec FlashInfer...")
llm = LLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    limit_mm_per_prompt={"image": 2}
)

# Paramètres de génération
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Test simple (texte uniquement)
prompts = [
    "Quelle est la capitale de la France?",
    "Explique-moi comment fonctionne un GPU en une phrase.",
]

print("\n=== Test 1: Génération de texte ===")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Réponse: {generated_text}")

print("\n✅ Test réussi! vLLM fonctionne sur GB10 avec Qwen3-Omni!")
