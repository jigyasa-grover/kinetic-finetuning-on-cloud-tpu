import json
import os
import time
from pathlib import Path

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ["KERAS_BACKEND"] = "jax"

import kinetic

# Load SFT data locally (Kinetic serializes it into the remote job)
_data_path = Path(__file__).parent / "data.jsonl"
_pairs = [json.loads(line) for line in _data_path.read_text().splitlines() if line.strip()]
SFT_DATA = {
    "prompts": [p["prompt"] for p in _pairs],
    "responses": [p["response"] for p in _pairs],
}

# Test prompts the model has never seen
TEST_PROMPTS = [
    "What is the tallest building in the world?",
    "Who discovered gravity?",
    "How many continents are there?",
    "What is the currency of Japan?",
    "Who painted the Mona Lisa?",
    "What is the smallest country in the world?",
    "How far is the Moon from Earth?",
    "What is the freezing point of water?",
]


@kinetic.run(accelerator="v5litepod-1", capture_env_vars=["KAGGLE_API_TOKEN"])
def finetune(sft_data, test_prompts):
    import jax
    import keras_hub

    devices = jax.devices()
    print(f"JAX {jax.__version__} | {len(devices)}x {devices[0].device_kind}")

    # Load Gemma 3 1B
    gemma = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")

    # Fine-tune on Gen Z data
    print(f"\nFine-tuning on {len(sft_data['prompts'])} pairs...")
    t0 = time.time()
    gemma.fit(x=sft_data, batch_size=1)
    train_time = time.time() - t0
    print(f"   Training time: {train_time:.1f}s")

    # Generate on unseen prompts
    print("\nGenerating on unseen prompts...")
    generations = {}
    for p in test_prompts:
        generations[p] = gemma.generate(p, max_length=80)

    return {
        "device": devices[0].device_kind,
        "jax_version": jax.__version__,
        "num_pairs": len(sft_data["prompts"]),
        "train_time_s": train_time,
        "generations": generations,
    }


if __name__ == "__main__":
    print(f"Loaded {len(SFT_DATA['prompts'])} SFT pairs from data.jsonl")
    print(f"{len(TEST_PROMPTS)} unseen test prompts")
    print("Shipping to TPU via Kinetic...\n")

    result = finetune(SFT_DATA, TEST_PROMPTS)

    # Print results
    print(f"\n{'='*70}")
    print(f"Fine-tuned on {result['device']} (JAX {result['jax_version']})")
    print(f"{result['num_pairs']} training pairs")
    print(f"Training time: {result['train_time_s']:.1f}s")
    print(f"{'='*70}\n")

    print("Generations on unseen prompts:\n")
    for prompt in TEST_PROMPTS:
        print(f"Q: {prompt}")
        print(f"A: {result['generations'][prompt]}")
        print()
