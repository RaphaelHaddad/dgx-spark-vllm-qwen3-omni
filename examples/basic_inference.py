#!/usr/bin/env python3
"""
Basic vLLM Inference Example for DGX Spark
Demonstrates simple text generation using the vLLM Python API
"""

from vllm import LLM, SamplingParams

def main():
    # Initialize the model
    # Use a smaller model for testing, replace with your preferred model
    print("Loading model...")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )

    # Define prompts
    prompts = [
        "What is the NVIDIA DGX Spark?",
        "Explain the Blackwell GPU architecture in simple terms.",
        "Write a haiku about artificial intelligence."
    ]

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=100,
        stop=["</s>", "\n\n\n"]
    )

    # Generate responses
    print("\nGenerating responses...\n")
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for i, output in enumerate(outputs):
        print(f"{'='*60}")
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"{'-'*60}")
        print(f"Response: {output.outputs[0].text}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
