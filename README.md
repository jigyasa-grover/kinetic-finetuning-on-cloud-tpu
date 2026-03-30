# In my Kinetic era: Fine-tuning Gemma 3 to speak Gen Z on a Cloud TPU with one decorator 🤌🏻

We fine-tuned Google's Gemma 3 1B to respond in Gen Z slang using supervised fine-tuning (SFT) on just 30 prompt/response pairs. The entire job runs on a Cloud TPU v5 Lite, deployed with a single Python decorator using the *new* [Kinetic](https://github.com/keras-team/kinetic) framework from the Keras team. 

Zero Docker. Zero Kubernetes YAML. Just `@kinetic.run()`!

For a deep dive into the setup, technical architecture, and a candid look at where Kinetic excels versus its current limitations, read the full write-up [here](https://jigyasa-grover.github.io/KineticFinetuningOnCloudTPU).

![In my Kinetic era: Fine-tuning Gemma 3 to speak Gen Z on a Cloud TPU with one decorator 🤌🏻](https://github.com/jigyasa-grover/kinetic-finetuning-on-cloud-tpu/blob/main/cover.png)

> Huge thanks to the Google ML Developer team for organizing this TPU sprint.
> Google Cloud credits were provided for this project ✨ 
> _#BuildWithAI #BuildWithGemma #BuildWithTPU #AISprint #GemmaSprint #TPUSprint_

## Prerequisites

- Python 3.11+
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- A GCP project with billing enabled
- A [Kaggle account](https://www.kaggle.com/) with the [Gemma license accepted](https://www.kaggle.com/models/keras/gemma3)

## GCP & Credential Setup

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
```

### 2. Select or create a GCP project

```bash
# List existing projects
gcloud projects list

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 3. Get your Kaggle API token

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** section
2. Click **"Generate New Token"**
3. Copy the token string (starts with `KGAT_...`)

### 4. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` with your values:

```
KAGGLE_API_TOKEN=your_kaggle_api_token
KERAS_REMOTE_PROJECT=your-gcp-project-id
KERAS_REMOTE_ZONE=us-central1-a
KERAS_REMOTE_CLUSTER=kinetic-cluster
```

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/jigyasa-grover/kinetic-finetuning-on-cloud-tpu.git
cd kinetic-finetuning-on-cloud-tpu
python3 -m venv .venv && source .venv/bin/activate
pip install "keras-kinetic[cli]"

# 2. Source credentials
source .env

# 3. Provision TPU infrastructure (one-time, ~5 min)
kinetic up --project=$KERAS_REMOTE_PROJECT --accelerator=v5litepod-1 --yes

# 4. Fine-tune Gemma on TPU
python finetune.py

# 5. Clean up (important — avoids idle cluster costs!)
kinetic down --yes
```

## Project Structure

```
kinetic-finetuning-on-cloud-tpu/
├── finetune.py        # Fine-tuning script (~100 lines)
├── data.jsonl         # 30 Gen Z SFT prompt/response pairs
├── .env.example       # Credential template
├── requirements.txt   # Remote container dependencies
```

## License

Apache 2.0
