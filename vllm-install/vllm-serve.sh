#!/bin/bash
# Script de démarrage du serveur vLLM Qwen3-Omni
# Lance le serveur en background et affiche la progression du chargement
# Usage: ./vllm-serve.sh [text|multimodal]
#   text (défaut) : Lance le serveur texte uniquement
#   multimodal    : Lance le serveur avec support audio/image

MODE="${1:-text}"  # Par défaut: mode texte
INSTALL_DIR="/home/hci-ai/Documents/vllm-mvp-omni-spark/dgx-spark-vllm-qwen3-omni/vllm-install"
TIMEOUT=600  # 10 minutes

# ============================================================
# CONFIGURATION DES CHEMINS — modifier uniquement ici
# ============================================================
export VLLM_INSTALL_DIR="$INSTALL_DIR"
export VLLM_MODEL_PATH="/home/hci-ai/Documents/models/models/Qwen3-Omni-30B-A3B-Instruct/"
# ============================================================

# Sélection du serveur et log selon le mode
if [ "$MODE" = "multimodal" ]; then
    SERVER_SCRIPT="server-qwen3-omni-multimodal.py"
    LOG_FILE="$INSTALL_DIR/vllm-server-multimodal.log"
    MODE_NAME="Multimodal (Audio + Image + Text)"
else
    SERVER_SCRIPT="server-qwen3-omni.py"
    LOG_FILE="$INSTALL_DIR/vllm-server.log"
    MODE_NAME="Text Only"
fi

cd "$INSTALL_DIR" || exit 1

echo "🚀 Démarrage du serveur vLLM Qwen3-Omni-30B..."
echo "📍 Répertoire: $INSTALL_DIR"
echo "🎯 Mode: $MODE_NAME"
echo "📝 Log: $LOG_FILE"
echo ""

# CRITIQUE : Arrêter les anciens processus ET vérifier que le GPU est libre
echo "🔍 Vérification des processus existants..."
./vllm-stop.sh

# Charger l'environnement et lancer le serveur
source vllm_env.sh
nohup python "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "🔧 Processus serveur lancé (PID: $SERVER_PID)"
echo "⏳ Chargement du modèle en cours (cela prend ~6-7 minutes)..."
echo ""

# Monitorer la progression du chargement
START_TIME=$(date +%s)
LAST_PROGRESS=""

while true; do
    ELAPSED=$(($(date +%s) - START_TIME))
    
    # Timeout après 7 minutes
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo ""
        echo "⏱️  Timeout atteint (7 minutes)"
        echo "🔍 Vérification de l'état du serveur..."
        break
    fi
    
    # Vérifier si le processus est toujours actif
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "❌ Le processus serveur s'est arrêté !"
        echo "📋 Dernières lignes du log :"
        tail -30 "$LOG_FILE"
        exit 1
    fi
    
    # Extraire la progression du chargement des shards
    if [ -f "$LOG_FILE" ]; then
        PROGRESS=$(grep "Loading safetensors checkpoint shards:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '\d+%')
        SHARD_INFO=$(grep "Loading safetensors checkpoint shards:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '\d+/\d+')
        
        if [ -n "$PROGRESS" ] && [ "$PROGRESS" != "$LAST_PROGRESS" ]; then
            printf "\r🔄 Chargement: $PROGRESS complété | Shards: $SHARD_INFO | Temps écoulé: ${ELAPSED}s     "
            LAST_PROGRESS="$PROGRESS"
        fi
        
        # Vérifier si le serveur est prêt
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            echo ""
            echo ""
            echo "✅ Serveur prêt !"
            echo "🌐 API disponible sur: http://localhost:8000"
            echo "🏥 Health check: http://localhost:8000/health"
            echo ""
            curl -s http://localhost:8000/health | python3 -m json.tool
            echo ""
            echo "📊 Processus actif:"
            ps aux | grep "server-qwen3-omni.py" | grep -v grep
            exit 0
        fi
        
        # Vérifier les erreurs critiques
        if grep -q "ERROR.*startup failed\|out of memory\|CUDA error" "$LOG_FILE" 2>/dev/null; then
            echo ""
            echo "❌ Erreur critique détectée !"
            echo "📋 Log d'erreur :"
            grep -A 5 "ERROR\|CUDA error" "$LOG_FILE" | tail -20
            exit 1
        fi
    fi
    
    sleep 2
done

# Vérification finale
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Serveur opérationnel"
else
    echo "⚠️  Le serveur ne répond pas encore. Consultez: tail -f $LOG_FILE"
fi
