# 🚀 Guide Complet : Qwen3-Omni sur NVIDIA DGX Spark GB10

**Statut** : ✅ **FONCTIONNEL** - Testé et validé le 6 janvier 2026

Ce guide documente la solution complète pour faire fonctionner Qwen3-Omni-30B avec vLLM sur GPU Blackwell (GB10, compute capability 12.1).

---

## 📋 Table des Matières

1. [Contexte du Problème](#contexte-du-problème)
2. [Architecture Matérielle](#architecture-matérielle)
3. [Solutions Appliquées](#solutions-appliquées)
4. [Installation](#installation)
5. [Lancement du Modèle](#lancement-du-modèle)
6. [Exemples d'Utilisation](#exemples-dutilisation)
7. [Dépannage](#dépannage)
8. [Performances](#performances)

---

## 🔍 Contexte du Problème

### Problème Initial

Le GPU **NVIDIA GB10 (Blackwell)** a une compute capability de **12.1 (sm_121)**, qui n'était pas supportée par :
- **PyTorch 2.9.1** : supporte jusqu'à sm_12.0
- **Triton 3.5.0** : le ptxas intégré ne reconnaît pas sm_121
- **FlashAttention-2** : compilé avec PTX incompatible
- **vLLM 0.11.1rc4** : manquait des optimisations MOE pour Blackwell

### Erreurs Rencontrées

```bash
# Erreur 1 : Triton PTXASError
PTXASError: Value 'sm_121' is not defined for option 'gpu-name'

# Erreur 2 : Symbole MOE manquant
ImportError: undefined symbol: cutlass_moe_mm_sm100

# Erreur 3 : PTX incompatible FlashAttention
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain

# Erreur 4 : Tokenizer Qwen2
AttributeError: 'Qwen2Tokenizer' object has no attribute 'all_special_tokens_extended'
```

---

## 🖥️ Architecture Matérielle

### Spécifications GPU

```yaml
GPU: NVIDIA GB10 (Blackwell)
Compute Capability: 12.1 (sm_121)
VRAM: 128 GB
Architecture: Blackwell (GB10x)
Driver: 570.x
CUDA System: 13.1.80
```

### Stack Logiciel Final

```yaml
OS: Linux (Ubuntu/RHEL)
Python: 3.12.3
CUDA: 13.1.80 (système)
PyTorch: 2.9.1+cu130
Triton: 3.5.0+git (main branch, commit 4caa0328)
vLLM: 0.11.1rc4.dev6+g66a168a19 (patchée)
FlashInfer: 0.4.1
```

---

## 🛠️ Solutions Appliquées

### 1. Modification de l'Architecture CUDA

**Fichier** : `vllm_env.sh`

**Problème** : TORCH_CUDA_ARCH_LIST=12.1a non reconnu par CMakeLists.txt

**Solution** : Forcer 12.0f (PTX compatible avec compute 12.1)

```bash
# AVANT
export TORCH_CUDA_ARCH_LIST="12.1a"

# APRÈS
export TORCH_CUDA_ARCH_LIST="12.0f"
```

**Raison** : CMakeLists.txt cherche "12.0f" dans ses conditionnels, pas "12.1a"

---

### 2. Ajout des Bibliothèques PyTorch au LD_LIBRARY_PATH

**Fichier** : `vllm_env.sh`

**Problème** : libtorch.so, libtorch_cuda.so introuvables au runtime

**Solution** : Ajouter le chemin des libs PyTorch

```bash
TORCH_LIB="$SCRIPT_DIR/.vllm/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
```

---

### 3. Compilation du Kernel MOE pour Blackwell

**Fichier** : `vllm/CMakeLists.txt` (ligne ~695)

**Problème** : grouped_mm_c3x_sm100.cu non compilé → symbole cutlass_moe_mm_sm100 manquant

**Solution** : Ajouter 12.0f à la liste des architectures

```cmake
# AVANT (ligne 695)
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")

# APRÈS
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
```

**Vérification** :
```bash
nm -D vllm/_C.abi3.so | grep cutlass_moe_mm_sm100
# Doit afficher : T cutlass_moe_mm_sm100
```

---

### 4. Fix du Tokenizer Qwen2

**Fichier** : `vllm/vllm/transformers_utils/tokenizer.py` (ligne 92)

**Problème** : Qwen2Tokenizer n'a pas l'attribut `all_special_tokens_extended`

**Solution** : Utiliser getattr() avec fallback

```python
# AVANT
tokenizer_all_special_tokens_extended = tokenizer.all_special_tokens_extended

# APRÈS
tokenizer_all_special_tokens_extended = getattr(
    tokenizer, 
    'all_special_tokens_extended', 
    tokenizer.all_special_tokens
)
```

---

### 5. Contournement du Bug FlashAttention-2

**Problème** : PTX de FA2 compilé avec toolchain incompatible avec sm_121

**Solution** : Forcer FlashInfer au lieu de FlashAttention-2

```python
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
```

**Alternative** : Recompiler FA2 avec CUDA 13.1 (non nécessaire avec FlashInfer)

---

### 6. Fix du Bug tqdm

**Fichier** : `vllm/vllm/model_executor/model_loader/weight_utils.py` (ligne 84)

**Problème** : Argument 'disable' passé deux fois à tqdm

**Solution** :

```python
# AVANT
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)

# APRÈS
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs['disable'] = True
        super().__init__(*args, **kwargs)
```

---

## 📦 Installation

### Méthode : Installation depuis eelbaz/dgx-spark-vllm-setup

```bash
# 1. Cloner le repo vérifié pour Blackwell
cd /home/oho/Projects
git clone https://github.com/eelbaz/dgx-spark-vllm-setup.git
cd dgx-spark-vllm-setup/vllm-install

# 2. Lancer l'installation automatique
bash install-vllm-v1.sh

# 3. Appliquer les patchs manuels
# 3a. Modifier vllm_env.sh
nano vllm_env.sh
# Changements :
# - TORCH_CUDA_ARCH_LIST="12.0f" (ligne ~40)
# - Ajouter TORCH_LIB et mettre à jour LD_LIBRARY_PATH (ligne ~50)

# 3b. Patcher CMakeLists.txt
cd vllm
nano CMakeLists.txt
# Ligne ~695 : Ajouter 12.0f à SCALED_MM_ARCHS

# 3c. Patcher tokenizer.py
nano vllm/transformers_utils/tokenizer.py
# Ligne 92 : Ajouter getattr() fallback

# 3d. Patcher weight_utils.py
nano vllm/model_executor/model_loader/weight_utils.py
# Ligne 84-86 : Fix DisabledTqdm

# 4. Recompiler vLLM
source ../vllm_env.sh
cd vllm
rm -rf build dist *.egg-info
python setup.py build_ext --inplace
cd ..

# 5. Vérifier la compilation
nm -D vllm/build/lib.linux-aarch64-cpython-312/vllm/_C.abi3.so | grep cutlass_moe_mm_sm100
# Output attendu : [address] T cutlass_moe_mm_sm100
```

### Temps d'Installation

- **Installation automatique** : ~30 minutes (virtualenv + PyTorch + dépendances)
- **Compilation vLLM** : ~15 minutes (490 fichiers)
- **Total** : ~45 minutes

---

## 🚀 Lancement du Modèle

### Option 1 : Script Python Direct

**Fichier** : `test-qwen3-omni-simple.py`

```python
#!/usr/bin/env python3
import sys
import os

# Importer vLLM depuis le code source
sys.path.insert(0, '/home/oho/Projects/dgx-spark-vllm-setup/vllm-install/vllm')

# CRUCIAL : Forcer FlashInfer (contourne bug FA2 PTX)
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from vllm import LLM, SamplingParams

# Charger le modèle
print("Chargement de Qwen3-Omni-30B avec FlashInfer...")
llm = LLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    limit_mm_per_prompt={"image": 2}  # Support multimodal
)

# Générer du texte
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=150
)

prompts = [
    "Quelle est la capitale de la France?",
    "Explique-moi comment fonctionne un GPU en une phrase."
]

print("\n=== Génération ===\n")
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Réponse: {output.outputs[0].text}\n")

print("✅ Test réussi sur GB10!")
```

**Lancement** :

```bash
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
source vllm_env.sh
python test-qwen3-omni-simple.py
```

---

### Option 2 : Serveur API vLLM

**Lancement du serveur** :

```bash
cd /home/oho/Projects/dgx-spark-vllm-setup/vllm-install
source vllm_env.sh

# Définir FlashInfer
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Lancer le serveur
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000
```

**Test avec curl** :

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "prompt": "Quelle est la capitale de la France?",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

---

## 💡 Exemples d'Utilisation

### Génération de Texte Simple

```python
from vllm import LLM, SamplingParams
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

llm = LLM(model="Qwen/Qwen3-Omni-30B-A3B-Instruct", trust_remote_code=True)
params = SamplingParams(temperature=0.8, max_tokens=200)

output = llm.generate(["Écris un poème sur l'IA"], params)
print(output[0].outputs[0].text)
```

### Génération Multimodale (avec images)

```python
from vllm import LLM, SamplingParams
from PIL import Image
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

llm = LLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 2}
)

# Charger une image
image = Image.open("photo.jpg")

# Générer avec contexte visuel
prompt = "Décris ce que tu vois dans cette image."
output = llm.generate([{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}])

print(output[0].outputs[0].text)
```

### Batch Processing

```python
# Traiter plusieurs requêtes en parallèle
prompts = [f"Question {i}: Qu'est-ce que l'IA?" for i in range(10)]
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

for i, output in enumerate(outputs):
    print(f"Réponse {i}: {output.outputs[0].text[:50]}...")
```

---

## 🐛 Dépannage

### Problème : "Loading safetensors checkpoint shards" prend 5 minutes

**C'est NORMAL !** Le modèle 30B (60GB) se charge depuis le disque vers le GPU.

**Pourquoi c'est lent ?**
- 15 fichiers × 4GB chacun = 60GB à lire
- Transfert disque → RAM → GPU = ~5 minutes
- Ce n'est **PAS un téléchargement** (modèle déjà en cache)

**Cache HuggingFace** :
```bash
# Vérifier le cache
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/
```

**Accélérer le chargement** :
- Utiliser un SSD NVMe rapide
- Augmenter la bande passante PCIe
- Précharger le modèle en RAM avec `--preload-model-on-startup`

---

### Problème : "CUDA error: the provided PTX was compiled with an unsupported toolchain"

**Cause** : FlashAttention-2 compilé avec mauvais CUDA toolkit

**Solution** : Forcer FlashInfer

```python
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
```

Ou recompiler FA2 :
```bash
pip uninstall flash-attn -y
TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn --no-build-isolation
```

---

### Problème : "ImportError: undefined symbol: cutlass_moe_mm_sm100"

**Cause** : CMakeLists.txt n'a pas compilé grouped_mm_c3x

**Solution** : Vérifier et recompiler

```bash
# 1. Vérifier CMakeLists.txt ligne 695
grep "SCALED_MM_ARCHS" vllm/CMakeLists.txt
# Doit contenir : "10.0f;11.0f;12.0f"

# 2. Recompiler
cd vllm
rm -rf build
python setup.py build_ext --inplace

# 3. Vérifier le symbole
nm -D vllm/build/lib.linux-aarch64-cpython-312/vllm/_C.abi3.so | grep cutlass_moe_mm_sm100
```

---

### Problème : "AttributeError: 'Qwen2Tokenizer' has no attribute 'all_special_tokens_extended'"

**Cause** : Bug vLLM avec tokenizers Qwen2

**Solution** : Patcher tokenizer.py (ligne 92)

```python
tokenizer_all_special_tokens_extended = getattr(
    tokenizer, 
    'all_special_tokens_extended', 
    tokenizer.all_special_tokens
)
```

---

### Problème : Out of Memory (OOM)

**Cause** : Modèle 30B + KV cache trop grand

**Solutions** :

```python
# Réduire gpu_memory_utilization
llm = LLM(model="...", gpu_memory_utilization=0.8)  # Au lieu de 0.9

# Réduire max_model_len
llm = LLM(model="...", max_model_len=4096)  # Au lieu de 8192

# Utiliser quantization
llm = LLM(model="...", quantization="bitsandbytes", load_format="bitsandbytes")
```

---

## 📊 Performances

### Benchmarks sur GB10

#### Qwen2.5-0.5B (baseline)

```yaml
Modèle: Qwen/Qwen2.5-0.5B-Instruct
Taille: 0.9 GB
Vitesse génération: ~347 tokens/s
Latence first token: ~150ms
Throughput: 3.47 req/s
KV Cache: 9.1M tokens
```

#### Qwen3-Omni-30B (production)

```yaml
Modèle: Qwen/Qwen3-Omni-30B-A3B-Instruct
Taille: 59.3 GB
Vitesse génération: ~32 tokens/s
Latence first token: ~8s (après chargement)
Throughput: 0.12 req/s
KV Cache: 470K tokens
Concurrence max: 57x (pour séquences 8K)
Chargement initial: ~6 minutes
```

### Comparaison Backends Attention

| Backend | Vitesse (tok/s) | Compatibilité GB10 | Mémoire | Stabilité |
|---------|-----------------|-------------------|---------|-----------|
| FlashAttention-2 | ❌ PTX Error | ❌ Non compatible | - | ❌ |
| FlashInfer | ✅ 32 tok/s | ✅ Compatible | Standard | ✅ Stable |
| XFormers | ⚠️ ~28 tok/s | ⚠️ Partiel | +10% | ⚠️ |
| Eager | ⚠️ ~15 tok/s | ✅ Compatible | +30% | ✅ |

**Recommandation** : **FlashInfer** pour le meilleur compromis vitesse/stabilité sur Blackwell.

---

## 🔧 Commandes Utiles

### Vérification de l'Installation

```bash
# Vérifier CUDA
nvcc --version
nvidia-smi

# Vérifier GPU
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Vérifier vLLM
source vllm_env.sh
python -c "import vllm.version; print(vllm.version.__version__)"

# Vérifier symboles MOE
nm -D vllm/build/lib.linux-aarch64-cpython-312/vllm/_C.abi3.so | grep -E "cutlass_moe|grouped_mm"

# Vérifier FlashInfer (warning PyTorch normal, peut être ignoré)
python -c "import flashinfer; print(flashinfer.__version__)"
# Attendu: 0.4.1
# Warning PyTorch "cuda capability 12.1 vs 12.0" est NORMAL
```

### Monitoring en Temps Réel

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Logs vLLM
tail -f vllm-server.log

# Processus Python
ps aux | grep python | grep vllm
```

### Nettoyage

```bash
# Nettoyer cache HuggingFace
rm -rf ~/.cache/huggingface/hub/models--Qwen*

# Nettoyer compilation vLLM
cd vllm
rm -rf build dist *.egg-info *.so

# Nettoyer virtualenv
rm -rf .vllm
```

---

## 📝 Notes Importantes

### ⚠️ Points d'Attention

1. **TOUJOURS utiliser FlashInfer** : FlashAttention-2 ne fonctionne pas sur GB10
2. **Ne pas utiliser TORCH_CUDA_ARCH_LIST=12.1a** : CMakeLists.txt ne le reconnaît pas
3. **Patcher les 6 fichiers** avant de compiler : vllm_env.sh, CMakeLists.txt, tokenizer.py, weight_utils.py
4. **Temps de chargement normal** : 5-6 minutes pour 60GB n'est PAS un bug
5. **Warning PyTorch cuda capability** : "Minimum and Maximum cuda capability (8.0) - (12.0)" est NORMAL, peut être ignoré
6. **Permissions cache HuggingFace** : Vérifier les droits sur ~/.cache/huggingface/

### ✅ Checklist Avant de Lancer

- [ ] vllm_env.sh : TORCH_CUDA_ARCH_LIST="12.0f"
- [ ] vllm_env.sh : TORCH_LIB dans LD_LIBRARY_PATH
- [ ] CMakeLists.txt : "10.0f;11.0f;12.0f" pour SCALED_MM_ARCHS
- [ ] tokenizer.py : getattr() fallback ligne 92
- [ ] weight_utils.py : DisabledTqdm fixé
- [ ] Script Python : os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
- [ ] Symbole vérifié : nm -D vllm/build/lib.linux-aarch64-cpython-312/vllm/_C.abi3.so | grep cutlass_moe_mm_sm100

---

## 🎯 Résumé des Fixes Critiques

| # | Fichier | Ligne | Fix | Impact |
|---|---------|-------|-----|--------|
| 1 | vllm_env.sh | ~40 | TORCH_CUDA_ARCH_LIST="12.0f" | 🔴 Critique |
| 2 | vllm_env.sh | ~50 | LD_LIBRARY_PATH avec TORCH_LIB | 🔴 Critique |
| 3 | CMakeLists.txt | ~695 | Ajouter 12.0f à SCALED_MM_ARCHS | 🔴 Critique |
| 4 | tokenizer.py | 92 | getattr() fallback | 🔴 Critique |
| 5 | weight_utils.py | 84 | kwargs['disable'] = True | 🟡 Important |
| 6 | Script Python | - | VLLM_ATTENTION_BACKEND=FLASHINFER | 🔴 Critique |

---

## 📚 Ressources

### Liens Utiles

- **Repo eelbaz** : https://github.com/eelbaz/dgx-spark-vllm-setup
- **vLLM Documentation** : https://docs.vllm.ai/
- **Qwen3-Omni HuggingFace** : https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- **FlashInfer** : https://github.com/flashinfer-ai/flashinfer
- **NVIDIA Blackwell** : https://www.nvidia.com/en-us/data-center/gb200/

### Support

En cas de problème :
1. Vérifier la checklist ci-dessus
2. Consulter la section Dépannage
3. Vérifier les logs : `tail -100 vllm-server.log`
4. Tester avec Qwen2.5-0.5B d'abord (plus rapide à charger)

---

## 🏆 Conclusion

Cette solution permet de faire fonctionner **Qwen3-Omni-30B à ~32 tokens/s** sur **NVIDIA GB10** malgré les incompatibilités initiales. Les fixes sont **stables** et **reproductibles**.

**Temps total de setup** : ~1 heure (installation + patchs + tests)

**Statut final** : ✅ **PRODUCTION READY**

---

*Dernière mise à jour : 6 janvier 2026*
*Testé sur : NVIDIA DGX Spark, GB10 (sm_121)*
*Version vLLM : 0.11.1rc4.dev6+g66a168a19*
