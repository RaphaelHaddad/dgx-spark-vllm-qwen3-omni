#!/bin/bash
# Script de redémarrage du serveur vLLM
# Arrête proprement puis relance le serveur

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Redémarrage du serveur vLLM..."
echo ""

# Arrêter le serveur
echo "1️⃣  Arrêt du serveur existant..."
"$SCRIPT_DIR/vllm-stop.sh"

echo ""
echo "2️⃣  Démarrage du nouveau serveur..."
sleep 2

# Démarrer le serveur
"$SCRIPT_DIR/vllm-serve.sh"
