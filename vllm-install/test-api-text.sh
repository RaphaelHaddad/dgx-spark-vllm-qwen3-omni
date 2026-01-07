#!/bin/bash
# Test de l'API vLLM avec requête texte simple (compatible OpenAI)

URL="http://localhost:8000/v1/completions"

echo "=== Test API vLLM - Génération de texte ==="
echo "URL: $URL"
echo ""

curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "prompt": "Quelle est la capitale de la France? Réponds en une phrase.",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": false
    }' | python -m json.tool

echo ""
echo "✅ Test terminé"
