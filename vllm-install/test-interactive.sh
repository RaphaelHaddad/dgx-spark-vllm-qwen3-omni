#!/bin/bash
# Script de test interactif pour l'API Qwen3-Omni

echo "🗣️  Test interactif de l'API Qwen3-Omni"
echo "========================================"
echo ""

# Fonction pour tester
test_chat() {
    local user_message="$1"
    local max_tokens="${2:-150}"
    
    echo "💬 Vous: $user_message"
    echo "⏳ Génération..."
    echo ""
    
    response=$(curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"Qwen/Qwen3-Omni-30B-A3B-Instruct\",
        \"messages\": [
          {\"role\": \"user\", \"content\": \"$user_message\"}
        ],
        \"max_tokens\": $max_tokens,
        \"temperature\": 0.7
      }")
    
    content=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$content" ]; then
        echo "🤖 Qwen3-Omni:"
        echo "$content"
        echo ""
        
        # Stats
        tokens=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Tokens: {data['usage']['total_tokens']} (prompt: {data['usage']['prompt_tokens']}, completion: {data['usage']['completion_tokens']})\")" 2>/dev/null)
        echo "📊 $tokens"
    else
        echo "❌ Erreur lors de la génération"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi
    echo ""
}

# Tests prédéfinis
echo "=== Tests automatiques ==="
echo ""

test_chat "Bonjour, comment vas-tu?" 50

test_chat "Explique-moi ce qu'est l'IA en une phrase" 80

test_chat "Raconte-moi une blague courte" 100

# Mode interactif
echo ""
echo "=== Mode interactif ==="
echo "Tapez vos questions (Ctrl+C pour quitter)"
echo ""

while true; do
    echo -n "💬 Vous: "
    read user_input
    
    if [ -z "$user_input" ]; then
        continue
    fi
    
    test_chat "$user_input" 200
done
