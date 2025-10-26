# vLLM Cluster Mode Setup for DGX Spark

This guide covers setting up multi-node vLLM deployment on DGX Spark systems using distributed inference.

## Prerequisites

- Multiple DGX Spark systems with vLLM installed (use `install.sh` on each node)
- All nodes on the same network with direct connectivity
- SSH access between nodes (passwordless SSH recommended)
- Same CUDA and vLLM versions across all nodes

## Architecture

```
┌─────────────────────┐
│   spark-alpha       │
│   (Master/Head)     │
│   - API Server      │
│   - Request Router  │
│   - Model Weights   │
└──────────┬──────────┘
           │
           ├─────────────────────┐
           │                     │
┌──────────▼──────────┐  ┌──────▼──────────┐
│   spark-omega       │  │   spark-gamma   │
│   (Worker 1)        │  │   (Worker 2)    │
│   - Inference       │  │   - Inference   │
│   - GPU Compute     │  │   - GPU Compute │
└─────────────────────┘  └─────────────────┘
```

## Step 1: Install vLLM on All Nodes

Run the installer on each node:

```bash
# On spark-alpha (master)
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash

# On spark-omega (worker 1)
ssh spark-omega.local
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash

# On spark-gamma (worker 2)
ssh spark-gamma.local
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
```

## Step 2: Configure Network Settings

Ensure all nodes can communicate on the required ports:

- **8000**: vLLM API server (master only)
- **29500**: PyTorch distributed backend (all nodes)
- **Random ports**: Ray cluster communication

Open firewall if needed:

```bash
# On all nodes
sudo ufw allow 8000/tcp
sudo ufw allow 29500/tcp
sudo ufw allow 6379/tcp   # Ray GCS
sudo ufw allow 8265/tcp   # Ray Dashboard
```

## Step 3: Set Up Passwordless SSH (Optional but Recommended)

```bash
# On master node
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Copy to worker nodes
ssh-copy-id spark-omega.local
ssh-copy-id spark-gamma.local

# Verify
ssh spark-omega.local "echo 'Connection successful'"
ssh spark-gamma.local "echo 'Connection successful'"
```

## Step 4: Start Ray Cluster

### On Master Node (spark-alpha)

```bash
# Assuming vllm-install is in your home directory
source ~/vllm-install/vllm_env.sh

# Start Ray head node
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=1

# Note the output: "To connect to this Ray cluster, use: ray start --address='MASTER_IP:6379'"
```

### On Worker Nodes (spark-omega, spark-gamma)

```bash
source ~/vllm-install/vllm_env.sh

# Replace MASTER_IP with spark-alpha's IP address
ray start --address='MASTER_IP:6379' --num-gpus=1
```

Verify cluster status:

```bash
ray status
```

You should see all nodes listed.

## Step 5: Start vLLM with Tensor Parallelism

### Method 1: Tensor Parallelism (Recommended for Large Models)

Tensor parallelism splits model layers across multiple GPUs.

```bash
# On master node
source ~/vllm-install/vllm_env.sh

vllm serve \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

This will automatically distribute the model across 2 GPUs in the Ray cluster.

### Method 2: Pipeline Parallelism

Pipeline parallelism splits model stages across GPUs.

```bash
vllm serve \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

### Method 3: Combined Parallelism

For very large models, combine tensor and pipeline parallelism:

```bash
vllm serve \
  --model "meta-llama/Llama-3.1-405B-Instruct" \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

## Step 6: Test Cluster Inference

```bash
# Test from master node
curl http://localhost:8000/v1/models

# Test from external machine
curl http://spark-alpha.local:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "Explain distributed inference in 3 sentences.",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Step 7: Monitor Cluster

### Ray Dashboard

Access at: http://spark-alpha.local:8265

Shows:
- Node status and resources
- Task execution
- GPU utilization
- Memory usage

### vLLM Metrics

```bash
# On master node
tail -f ~/vllm-install/vllm-server.log

# Check GPU usage across cluster
ray exec 'nvidia-smi'
```

### System Monitoring

```bash
# Check Ray cluster status
ray status

# Monitor GPU usage on specific node
ssh spark-omega.local nvidia-smi -l 1
```

## Troubleshooting

### Workers Not Connecting

**Problem**: Workers can't connect to Ray head node

**Solutions**:
1. Check firewall: `sudo ufw status`
2. Verify head node IP: `ray status` on master
3. Check network connectivity: `ping spark-alpha.local`
4. Ensure same Ray version on all nodes: `ray --version`

### OOM Errors with Large Models

**Problem**: Out of memory when loading large models

**Solutions**:
1. Increase tensor parallelism: `--tensor-parallel-size 4`
2. Reduce memory utilization: `--gpu-memory-utilization 0.8`
3. Enable CPU offloading: `--cpu-offload-gb 8`
4. Use quantization: `--quantization awq` or `--quantization gptq`

### Model Loading Hangs

**Problem**: Model download/loading takes forever

**Solutions**:
1. Pre-download model on all nodes:
   ```bash
   # On each node
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-70B-Instruct')"
   ```
2. Use shared storage (NFS) for model cache
3. Check network bandwidth between nodes

### Uneven GPU Utilization

**Problem**: Some GPUs idle while others maxed out

**Solutions**:
1. Verify tensor parallel configuration
2. Check Ray resource allocation: `ray status`
3. Ensure balanced request distribution
4. Monitor with: `ray exec 'nvidia-smi'`

## Advanced Configuration

### Custom Ray Resources

Assign custom resources to nodes for fine-grained control:

```bash
# On worker with high memory
ray start --address='MASTER_IP:6379' \
  --num-gpus=1 \
  --resources='{"highmem": 1}'

# Use in vLLM
vllm serve --model "..." --placement-group-resources='{"highmem": 1}'
```

### Distributed Model Cache

Share model weights via NFS to avoid redundant downloads:

```bash
# On NFS server (e.g., master)
sudo apt install nfs-kernel-server
echo "$HOME/.cache/huggingface *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -a

# On workers
sudo apt install nfs-common
sudo mkdir -p $HOME/.cache/huggingface
sudo mount spark-alpha.local:$HOME/.cache/huggingface $HOME/.cache/huggingface
```

### Load Balancing with nginx

For production deployments, use nginx to load balance across multiple vLLM instances:

```nginx
upstream vllm_cluster {
    least_conn;
    server spark-alpha.local:8000;
    server spark-omega.local:8000;
    server spark-gamma.local:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_cluster;
        proxy_set_header Host $host;
    }
}
```

## Cluster Management Scripts

### Start Cluster

Create `start-cluster.sh`:

```bash
#!/bin/bash
# Start Ray cluster on all nodes

ssh spark-alpha.local "source ~/vllm-install/vllm_env.sh && ray start --head --port=6379"
sleep 5

MASTER_IP=$(ssh spark-alpha.local "hostname -I | awk '{print \$1}'")

ssh spark-omega.local "source ~/vllm-install/vllm_env.sh && ray start --address='${MASTER_IP}:6379'"
ssh spark-gamma.local "source ~/vllm-install/vllm_env.sh && ray start --address='${MASTER_IP}:6379'"

echo "Cluster started. Check status with: ray status"
```

### Stop Cluster

Create `stop-cluster.sh`:

```bash
#!/bin/bash
# Stop Ray cluster on all nodes

for node in spark-alpha.local spark-omega.local spark-gamma.local; do
    echo "Stopping Ray on $node..."
    ssh $node "ray stop --force"
done

echo "Cluster stopped."
```

## Performance Tuning

### For Maximum Throughput

```bash
vllm serve \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --tensor-parallel-size 2 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95
```

### For Low Latency

```bash
vllm serve \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --tensor-parallel-size 2 \
  --max-num-seqs 32 \
  --disable-log-requests
```

## References

- [vLLM Distributed Inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [Ray Cluster Setup](https://docs.ray.io/en/latest/cluster/getting-started.html)
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)

## Support

For issues specific to DGX Spark cluster setup, please open an issue on GitHub.
