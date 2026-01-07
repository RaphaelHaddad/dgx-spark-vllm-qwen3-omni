#!/bin/bash
# Script de vérification de l'état du serveur vLLM
# Affiche l'état complet : processus, GPU, API

INSTALL_DIR="/home/oho/Projects/dgx-spark-vllm-setup/vllm-install"

echo "📊 État du serveur vLLM Qwen3-Omni"
echo "=================================="
echo ""

# 1. Vérifier les processus
echo "🔧 Processus Python vLLM:"
PROCESS_COUNT=$(ps aux | grep -E "server-qwen3-omni.py" | grep -v grep | wc -l)
if [ $PROCESS_COUNT -gt 0 ]; then
    ps aux | grep "server-qwen3-omni.py" | grep -v grep | awk '{printf "  PID: %s | CPU: %s%% | MEM: %s%% | CMD: %s\n", $2, $3, $4, $11}'
    echo "  ✅ $PROCESS_COUNT processus actif(s)"
else
    echo "  ❌ Aucun processus actif"
fi

echo ""

# 2. Vérifier la mémoire GPU
echo "🎮 Mémoire GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s (%s): %s / %s | Utilisation: %s\n", $1, $2, $3, $4, $5}'
else
    echo "  ⚠️  nvidia-smi non disponible"
fi

echo ""

# 3. Health check API
echo "🏥 API Health Check:"
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$HEALTH_RESPONSE" ]; then
    echo "  ✅ API accessible sur http://localhost:8000"
    echo "  📝 Réponse:"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null | sed 's/^/    /'
else
    echo "  ❌ API non accessible"
fi

echo ""

# 4. Log récent
echo "📋 Dernières lignes du log (non-erreurs):"
if [ -f "$INSTALL_DIR/vllm-server.log" ]; then
    tail -10 "$INSTALL_DIR/vllm-server.log" | grep -v "Permission denied" | sed 's/^/  /'
else
    echo "  ⚠️  Fichier log non trouvé"
fi

echo ""
echo "💡 Commandes utiles:"
echo "  - Voir les logs: tail -f $INSTALL_DIR/vllm-server.log"
echo "  - Arrêter: ./vllm-stop.sh"
echo "  - Démarrer: ./vllm-serve.sh"
echo "  - Redémarrer: ./vllm-restart.sh"
