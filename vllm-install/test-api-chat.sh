#!/bin/bash
# Test de l'API Chat (format OpenAI /v1/chat/completions)

URL="http://localhost:8000/v1/chat/completions"

echo "=== Test API Chat (format OpenAI) ==="
echo "URL: $URL"
echo ""

curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {"role": "system", "content": "Tu es un assistant IA intelligent."},
            {"role": "user", "content": "Explique-moi comment fonctionne un GPU en 2 phrases."}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }' | python -m json.tool

echo ""
echo "✅ Test terminé"
