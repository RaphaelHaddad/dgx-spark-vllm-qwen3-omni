#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/home/oho/Projects/dgx-spark-vllm-setup/vllm-install/vllm')

# Force FlashInfer backend (FA2 PTX incompatible with sm_121)
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from vllm import LLM, SamplingParams

print("Chargement du modèle Qwen2.5-0.5B-Instruct avec FlashInfer...")
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
    max_model_len=2048,
    disable_log_stats=True,
    enforce_eager=False
)

print("\n✅ Modèle chargé avec succès !")

# Test simple
prompts = [
    "Quelle est la capitale de la France?",
    "Explique en une phrase ce qu'est un GPU."
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

print("\n=== Génération de texte ===\n")
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Réponse: {output.outputs[0].text}\n")

print("✅ Test terminé avec succès sur GB10!")
