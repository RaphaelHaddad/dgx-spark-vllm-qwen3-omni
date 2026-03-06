#!/bin/bash
# Script d'arrêt complet de vLLM
# Tue TOUS les processus liés à vLLM et nettoie proprement

echo "🛑 Arrêt de tous les processus vLLM..."

# Tuer tous les processus Python liés au serveur
pkill -9 -f "server-qwen3-omni" 2>/dev/null
pkill -9 -f "python.*vllm" 2>/dev/null
pkill -9 -f "vllm.*worker" 2>/dev/null

# CRITIQUE : Tuer aussi les processus EngineCore (fils de vLLM v1)
pkill -9 -f "EngineCore" 2>/dev/null

# Tuer tous les PIDs utilisant le GPU
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
if [ -n "$GPU_PIDS" ]; then
    echo "🔪 Nettoyage GPU : $GPU_PIDS"
    echo "$GPU_PIDS" | xargs -r kill -9 2>/dev/null
fi

sleep 2

# Vérifier qu'il ne reste plus de processus
REMAINING=$(ps aux | grep -E "server-qwen3|vllm.*worker|EngineCore" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ Tous les processus vLLM sont arrêtés"
else
    echo "⚠️  Il reste $REMAINING processus actifs :"
    ps aux | grep -E "server-qwen3|vllm.*worker|EngineCore" | grep -v grep
fi

# Afficher l'état de la mémoire GPU avec détail des processus
echo ""
echo "📊 État de la mémoire GPU :"
GPU_USAGE=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$GPU_USAGE" ]; then
    echo "✅ Aucun processus n'utilise le GPU"
else
    # Ignore system/display processes using < 500 MiB (e.g. display manager, pid 176)
    SIGNIFICANT=$(echo "$GPU_USAGE" | awk -F',' '{gsub(/ /, "", $2); if ($2+0 >= 500) print}' )
    if [ -z "$SIGNIFICANT" ]; then
        echo "✅ Seuls des processus système mineurs utilisent le GPU (< 500 MiB), on continue"
    else
        echo "⚠️  ATTENTION : Des processus significatifs utilisent encore le GPU :"
        echo "$SIGNIFICANT"
        echo ""
        echo "❌ ÉCHEC : Impossible de continuer tant que le GPU n'est pas libéré"
        exit 1
    fi
fi
