<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GKE Deployment with Foundational RAG

Deploy AI-Q from source on a GKE cluster where Foundational RAG (fRAG) is already running. This guide builds custom Docker images that include the **enterprise knowledge search** tool, pushes them to Google Artifact Registry, and deploys via the source Helm chart.

> **Why build from source?** The published NGC images are built from `main` and do not include the `enterprise_knowledge_search` NAT plugin. Swapping the config file alone is not enough — the plugin must be installed in the image for the backend to load it.

## Prerequisites

- GKE cluster with `kubectl` and `helm` v3.x configured
- Docker installed (`linux/amd64` platform support required if building on ARM/Apple Silicon)
- `gcloud` CLI authenticated with access to the GKE cluster
- fRAG already deployed in the same cluster (with known service names, namespace, and ports)
- API keys: `NVIDIA_API_KEY`, `TAVILY_API_KEY` (optionally `SERPER_API_KEY` for paper search)

## Step 1: Clone the Repository

```bash
git clone https://github.com/pastorsj/aiq-research-assistant.git
cd aiq-research-assistant
git checkout feature/enterprise-knowledge-search
```

## Step 2: Set Up Google Artifact Registry

```bash
export GCP_PROJECT=your-gcp-project
export GCP_REGION=us-central1
export AR_REPO=aiq-images
export REGISTRY=${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}

# Create the repository (skip if it already exists)
gcloud artifacts repositories create $AR_REPO \
  --repository-format=docker \
  --location=$GCP_REGION \
  --description="AI-Q custom images"

# Configure Docker authentication
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
```

## Step 3: Build and Push Docker Images

From the repository root:

```bash
# Backend (includes enterprise_knowledge_search)
docker build --platform linux/amd64 \
  -f deploy/Dockerfile \
  --target release \
  -t ${REGISTRY}/aiq-agent:custom \
  .

# Frontend (Next.js UI)
docker build --platform linux/amd64 \
  -f frontends/ui/deploy/Dockerfile \
  -t ${REGISTRY}/aiq-frontend:custom \
  frontends/ui

# Push both
docker push ${REGISTRY}/aiq-agent:custom
docker push ${REGISTRY}/aiq-frontend:custom
```

> **Note:** `--platform linux/amd64` is required when building on Apple Silicon. GKE nodes are AMD64.

## Step 4: Confirm fRAG Service Endpoints

Look up the fRAG service names and ports in your cluster:

```bash
kubectl get svc -n rag    # adjust namespace to match your fRAG deployment
```

Typical services:

| Service | Port | Internal DNS |
|---------|------|--------------|
| `rag-server` | 8081 | `http://rag-server.rag.svc.cluster.local:8081/v1` |
| `ingestor-server` | 8082 | `http://ingestor-server.rag.svc.cluster.local:8082/v1` |

Use the actual values from your cluster in the Helm command below.

## Step 5: Create Namespace and Secrets

```bash
kubectl create namespace ns-aiq --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic aiq-credentials -n ns-aiq \
  --from-literal=NVIDIA_API_KEY="$NVIDIA_API_KEY" \
  --from-literal=TAVILY_API_KEY="$TAVILY_API_KEY" \
  --from-literal=DB_USER_NAME="aiq" \
  --from-literal=DB_USER_PASSWORD="aiq_dev"  # pragma: allowlist secret
```

### Image Pull Secrets (if needed)

GKE nodes in the **same project** as the Artifact Registry typically have read access by default. If they don't (e.g., cross-project), create a pull secret:

```bash
kubectl create secret docker-registry ar-pull-secret -n ns-aiq \
  --docker-server=${GCP_REGION}-docker.pkg.dev \
  --docker-username=oauth2accesstoken \
  --docker-password="$(gcloud auth print-access-token)"
```

Then add these flags to the Helm command in the next step:

```
--set 'aiq.apps.backend.imagePullSecrets[0].name=ar-pull-secret' \
--set 'aiq.apps.frontend.imagePullSecrets[0].name=ar-pull-secret'
```

## Step 6: Deploy with Helm

```bash
cd deploy/helm
helm dependency update deployment-k8s/

helm upgrade --install aiq deployment-k8s/ \
  -n ns-aiq \
  --create-namespace \
  --wait --timeout 10m \
  --set aiq.project.cloud_provider=gke \
  --set aiq.apps.backend.image.repository=${REGISTRY}/aiq-agent \
  --set aiq.apps.backend.image.tag=custom \
  --set aiq.apps.backend.image.pullPolicy=Always \
  --set aiq.apps.frontend.image.repository=${REGISTRY}/aiq-frontend \
  --set aiq.apps.frontend.image.tag=custom \
  --set aiq.apps.frontend.image.pullPolicy=Always \
  --set aiq.apps.backend.env.CONFIG_FILE=configs/config_web_frag.yml \
  --set aiq.apps.backend.env.RAG_SERVER_URL=http://rag-server.rag.svc.cluster.local:8081/v1 \
  --set aiq.apps.backend.env.RAG_INGEST_URL=http://ingestor-server.rag.svc.cluster.local:8082/v1
```

### Alternative: Values File

Create `gke-frag-values.yaml` for easier management:

```yaml
aiq:
  project:
    cloud_provider: gke
  apps:
    backend:
      image:
        repository: us-central1-docker.pkg.dev/YOUR_PROJECT/aiq-images/aiq-agent
        tag: custom
        pullPolicy: Always
      env:
        CONFIG_FILE: configs/config_web_frag.yml
        RAG_SERVER_URL: http://rag-server.rag.svc.cluster.local:8081/v1
        RAG_INGEST_URL: http://ingestor-server.rag.svc.cluster.local:8082/v1
    frontend:
      image:
        repository: us-central1-docker.pkg.dev/YOUR_PROJECT/aiq-images/aiq-frontend
        tag: custom
        pullPolicy: Always
```

```bash
helm upgrade --install aiq deployment-k8s/ \
  -n ns-aiq --create-namespace \
  --wait --timeout 10m \
  -f gke-frag-values.yaml
```

## Step 7: Verify

```bash
# Pod status (expect 3 pods: backend, frontend, postgres)
kubectl get pods -n ns-aiq

# Backend logs — look for enterprise_knowledge_search in loaded plugins
kubectl logs -n ns-aiq -l component=backend -f

# Health check
kubectl port-forward -n ns-aiq svc/aiq-backend 8000:8000 &
curl http://localhost:8000/health

# Access the UI — Enterprise Knowledge should appear in Data Sources
kubectl port-forward -n ns-aiq svc/aiq-frontend 3000:3000
# Open http://localhost:3000

# Test RAG connectivity from inside the backend pod
kubectl exec -n ns-aiq deployment/aiq-backend -- \
  python -c "import urllib.request; print(urllib.request.urlopen('http://rag-server.rag.svc.cluster.local:8081/v1/health').read())"
```

## Pre-populated Collections

The enterprise knowledge search tool queries **pre-existing collections** in fRAG's Milvus. The default config expects a `cybersecurity` collection. If it doesn't exist yet, populate it from inside the cluster:

```bash
# Option A: Port-forward the ingest service and run locally
kubectl port-forward -n rag svc/ingestor-server 8082:8082 &

uv run python scripts/populate_collection.py \
  data/cybersecurity cybersecurity \
  --backend foundational_rag \
  --rag-ingest-url http://localhost:8082/v1

# Option B: Run as a temporary pod inside the cluster
kubectl run populate-collections -n ns-aiq --rm -it \
  --image=${REGISTRY}/aiq-agent:custom \
  --restart=Never \
  --env="NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  -- python scripts/populate_collection.py \
    data/cybersecurity cybersecurity \
    --backend foundational_rag \
    --rag-ingest-url http://ingestor-server.rag.svc.cluster.local:8082/v1
```

## Rebuilding and Updating

```bash
# Rebuild with a new tag
docker build --platform linux/amd64 -f deploy/Dockerfile --target release \
  -t ${REGISTRY}/aiq-agent:v2 .
docker push ${REGISTRY}/aiq-agent:v2

# Update the deployment
helm upgrade aiq deploy/helm/deployment-k8s/ -n ns-aiq \
  --reuse-values \
  --set aiq.apps.backend.image.tag=v2

# Or with the same tag (pullPolicy=Always):
kubectl rollout restart deployment -n ns-aiq aiq-backend
```

## Uninstall

```bash
helm uninstall aiq -n ns-aiq
kubectl delete namespace ns-aiq    # removes all resources including secrets and PVCs
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImagePullBackOff` | GKE nodes can't pull from Artifact Registry | Create `ar-pull-secret` (see Step 5) or grant `artifactregistry.reader` to the node service account |
| Enterprise Knowledge not in Data Sources | Plugin not installed in image | Verify the Dockerfile includes `uv pip install -e ./sources/enterprise_knowledge_search` and rebuild |
| `CrashLoopBackOff` on backend | Missing secrets or bad config | Check `kubectl logs <pod> -n ns-aiq` and verify `aiq-credentials` has all required keys |
| RAG connection refused | Wrong service name, port, or namespace | Run `kubectl get svc -n rag` and update the RAG URLs |
| Empty enterprise search results | Collection doesn't exist in Milvus | Populate the collection (see [Pre-populated Collections](#pre-populated-collections)) |
| `paper_search_tool` errors | Missing `SERPER_API_KEY` | Add it to `aiq-credentials` or remove `paper_search_tool` from the config |
| CORS errors in browser | Frontend accessed via non-localhost origin | Update `cors.allow_origin_regex` in `config_web_frag.yml` |

## Architecture

```
GKE Cluster
  |
  +-- Namespace: rag (existing fRAG deployment)
  |     +-- rag-server (port 8081)       -- query endpoint
  |     +-- ingestor-server (port 8082)  -- ingestion endpoint
  |
  +-- Namespace: ns-aiq (this deployment)
        +-- aiq-backend (port 8000)      -- FastAPI + enterprise_knowledge_search
        |     CONFIG_FILE = configs/config_web_frag.yml
        |     RAG_SERVER_URL = http://rag-server.rag.svc.cluster.local:8081/v1
        |     RAG_INGEST_URL = http://ingestor-server.rag.svc.cluster.local:8082/v1
        |
        +-- aiq-frontend (port 3000)     -- Next.js UI
        |     BACKEND_URL = http://aiq-backend:8000
        |
        +-- aiq-postgres (port 5432)     -- PostgreSQL (jobs, checkpoints)
        |
        +-- Secret: aiq-credentials
              NVIDIA_API_KEY, TAVILY_API_KEY, DB_USER_NAME, DB_USER_PASSWORD
```
