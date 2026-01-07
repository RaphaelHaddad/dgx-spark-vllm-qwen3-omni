#!/bin/bash
# Script d'arrêt complet de vLLM
# Tue TOUS les processus liés à vLLM et nettoie proprement

echo "🛑 Arrêt de tous les processus vLLM..."

# Tuer tous les processus Python liés au serveur
pkill -9 -f "server-qwen3-omni.py" 2>/dev/null
pkill -9 -f "python.*vllm" 2>/dev/null
pkill -9 -f "vllm.*worker" 2>/dev/null

sleep 2

# Vérifier qu'il ne reste plus de processus
REMAINING=$(ps aux | grep -E "server-qwen3|vllm.*worker" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ Tous les processus vLLM sont arrêtés"
else
    echo "⚠️  Il reste $REMAINING processus actifs :"
    ps aux | grep -E "server-qwen3|vllm.*worker" | grep -v grep
fi

# Afficher l'état de la mémoire GPU
echo ""
echo "📊 État de la mémoire GPU :"
nvidia-smi | grep -A 1 "Memory-Usage"
