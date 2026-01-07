# 🚀 Lancement Rapide - Qwen3-Omni API Serveur

## 📌 Réponses à vos questions

### 1. Le modèle est-il actuellement chargé ?

**OUI**, le serveur est en cours de démarrage. Vérifiez avec :

```bash
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
tail -f vllm-server.log
```

Le chargement prend **~6 minutes** (60GB). Vous verrez :
```
Loading safetensors checkpoint shards: 100% | 15/15 [05:43<00:00]
✅ Serveur prêt!
```

### 2. Comment lancer le serveur ?

#### Méthode Simple (Recommandée)

```bash
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
source vllm_env.sh
python server-qwen3-omni.py
```

#### En arrière-plan

```bash
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
source vllm_env.sh
nohup python server-qwen3-omni.py > vllm-server.log 2>&1 &
```

#### Arrêter le serveur

```bash
pkill -f "python server-qwen3-omni.py"
```

### 3. Accès via URL (Compatible OpenAI API)

Une fois le serveur démarré (après ~6 min), l'API est disponible sur :

- **Base URL** : `http://localhost:8000`
- **Health Check** : `http://localhost:8000/health`
- **Completions** : `http://localhost:8000/v1/completions`
- **Chat** : `http://localhost:8000/v1/chat/completions`
- **Models** : `http://localhost:8000/v1/models`

---

## 🧪 Tests Disponibles

### Test 1 : Santé du serveur

```bash
curl http://localhost:8000/health
```

**Réponse attendue** :
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "backend": "FlashInfer"
}
```

### Test 2 : Génération de texte (Completions)

```bash
./test-api-text.sh
```

Ou manuellement :

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "prompt": "Quelle est la capitale de la France?",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool
```

### Test 3 : Chat (Format OpenAI)

```bash
./test-api-chat.sh
```

Ou manuellement :

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {"role": "system", "content": "Tu es un assistant IA intelligent."},
      {"role": "user", "content": "Explique-moi comment fonctionne un GPU."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }' | python -m json.tool
```

### Test 4 : Analyse Audio 🎵

**Important** : Vous devez avoir un fichier audio (WAV, MP3, etc.)

```bash
# Exemple avec un fichier audio
python test-api-audio.py /chemin/vers/votre/fichier.wav
```

**Format de la requête audio** :

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Transcris et analyse cet audio."
          },
          {
            "type": "audio_url",
            "audio_url": {
              "url": "data:audio/wav;base64,<BASE64_ENCODED_AUDIO>"
            }
          }
        ]
      }
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

**Génération d'un fichier audio de test** :

```bash
# Installer espeak (synthèse vocale)
sudo apt install espeak

# Générer un fichier WAV de test
espeak "Bonjour, ceci est un test audio pour Qwen3 Omni" \
  -w test-audio.wav --stdout | ffmpeg -i - test-audio.wav

# Tester l'analyse
python test-api-audio.py test-audio.wav
```

---

## 🔌 Utilisation comme Client OpenAI

Le serveur est **100% compatible avec l'API OpenAI**. Vous pouvez utiliser le client officiel :

### Python avec openai

```python
from openai import OpenAI

# Pointer vers votre serveur local
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Pas de clé nécessaire pour un serveur local
)

# Utiliser comme un modèle OpenAI classique
response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[
        {"role": "user", "content": "Quelle est la capitale de la France?"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/TypeScript avec openai

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy'  // Pas de clé nécessaire
});

const response = await client.chat.completions.create({
  model: 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
  messages: [
    { role: 'user', content: 'Quelle est la capitale de la France?' }
  ]
});

console.log(response.choices[0].message.content);
```

### cURL (exemple complet)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

---

## 📊 Monitoring

### Vérifier le statut

```bash
./check-server-status.sh
```

### Voir les logs en temps réel

```bash
tail -f vllm-server.log
```

### Voir les processus actifs

```bash
ps aux | grep "python server-qwen3-omni"
```

### Voir l'utilisation GPU

```bash
watch -n 1 nvidia-smi
```

---

## ⚡ Performances Attendues

| Métrique | Valeur |
|----------|--------|
| **Temps de chargement** | ~6 minutes (60GB) |
| **Vitesse génération** | ~32 tokens/s |
| **Latence first token** | ~8 secondes |
| **Mémoire GPU utilisée** | 59.3 GB (modèle) + 43.1 GB (KV cache) |
| **Contexte max** | 8192 tokens |
| **Support multimodal** | ✅ Texte, Audio, Images |

---

## 🐛 Dépannage Rapide

### Le serveur ne démarre pas

```bash
# Vérifier qu'aucun autre serveur n'utilise le port 8000
lsof -i :8000
pkill -f "python server-qwen3-omni"

# Relancer
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
source vllm_env.sh
python server-qwen3-omni.py
```

### Le chargement est bloqué

```bash
# Voir les dernières lignes du log
tail -50 vllm-server.log

# Si "Loading safetensors checkpoint shards" est affiché, c'est NORMAL
# Attendez les 6 minutes complètes
```

### Erreur "Connection refused"

Le serveur n'est pas encore prêt. Attendez le message :
```
✅ Serveur prêt!
```

### Out of Memory

Réduisez `gpu_memory_utilization` dans [server-qwen3-omni.py](server-qwen3-omni.py) :

```python
llm = LLM(
    model=MODEL,
    gpu_memory_utilization=0.8  # Au lieu de 0.9
)
```

---

## 📝 Fichiers Créés

| Fichier | Description |
|---------|-------------|
| `server-qwen3-omni.py` | ⭐ **Serveur principal** (API compatible OpenAI) |
| `test-api-text.sh` | Test génération texte simple |
| `test-api-chat.sh` | Test chat (format OpenAI) |
| `test-api-audio.py` | Test analyse audio |
| `check-server-status.sh` | Vérifier statut du serveur |
| `vllm-server.log` | Logs du serveur |

---

## 🎯 Commandes Essentielles

```bash
# Lancer le serveur
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
source vllm_env.sh
python server-qwen3-omni.py

# Tester (dans un autre terminal)
curl http://localhost:8000/health
./test-api-text.sh
./test-api-chat.sh

# Arrêter
pkill -f "python server-qwen3-omni"
```

---

## 📚 Ressources

- **Documentation complète** : [FUNCTIONAL_OHO_README.md](FUNCTIONAL_OHO_README.md)
- **API OpenAI** : https://platform.openai.com/docs/api-reference
- **vLLM Docs** : https://docs.vllm.ai/
- **Qwen3-Omni** : https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct

---

**Status** : ✅ **FONCTIONNEL** - Serveur démarré, chargement en cours (7% après 30s)

*Attendez ~6 minutes pour que le modèle soit complètement chargé avant de tester l'API.*
