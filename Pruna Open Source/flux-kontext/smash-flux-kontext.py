import os.path
import time
from io import BytesIO
from pathlib import Path

import modal


cuda_version = "12.6.0"
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
VARIANT = "dev"
model_id = f"black-forest-labs/FLUX.1-Kontext-dev"
model_cache_name = f"{model_id}-pruna-smashed"

# Build the CUDA development image
cuda_dev_image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
cuda_dev_image = cuda_dev_image.apt_install("git")            # ← ensure Git is available for VCS installs
cuda_dev_image = cuda_dev_image.pip_install(
            "pruna",
            "git+https://github.com/huggingface/diffusers.git",
            "transformers",
            "pillow",
            "huggingface_hub",
            pre=True,
        )

app = modal.App("flux-dev", image=cuda_dev_image)

with cuda_dev_image.imports():
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from pruna import SmashConfig
    from pruna import smash
    from pruna import PrunaModel
    from huggingface_hub import HfApi
    import torch
    import gc

MINUTES = 60  # seconds (original 10)


@app.cls(
    gpu="H100",  # fastest GPU on Modal
    scaledown_window=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts and caches
        "/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/model_cache": modal.Volume.from_name("model_cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name("inductor-cache", create_if_missing=True),
    },
)
class Model:
    compile: bool = (  # see section on torch.compile below for details
        modal.parameter(default=False)
    )

    @modal.enter()
    def enter(self):
        """
        Load the base model, run a one-off benchmark, then build / reload the
        optimised model.  Benchmark artefacts are stored in `self._bench`.
        The base model is deleted afterwards to free memory.
        """
        # ---------- common inputs used for the benchmark ------------------
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Using device:", torch.cuda.get_device_name(0))

        sample_prompt = "A playful cat wearing a blue wizard hat"
        sample_image_url = (
            "https://huggingface.co/datasets/huggingface/documentation-images/"
            "resolve/main/diffusers/cat.png"
        )
        sample_image = load_image(sample_image_url)

        # ---------- BASE model -------------------------------------------
        print("🔄 Loading base model …")
        t0 = time.time()
        base_model = FluxKontextPipeline.from_pretrained(
            model_id, cache_dir="/cache",
            token="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",
            torch_dtype=torch.float16,
        ).to("cuda")
        base_load_time = time.time() - t0
        print(f"✅ Base model loaded in {base_load_time:.2f}s")

        # Inference latency with base model
        t0 = time.time()
        base_model(
            image=sample_image, prompt=sample_prompt, guidance_scale=2.5
        )

        base_latency = time.time() - t0
        print(f"⏱️  Base inference latency: {base_latency:.2f}s")

        # ---------- OPTIMISED model --------------------------------------
        if not os.path.exists(f"/model_cache/{model_cache_name}"):
            save_the_model = True  # Only save after warmup
            print("🚀 Optimising model with Pruna Smash …")
            smash_cfg = SmashConfig()
            smash_cfg["compiler"] = "torch_compile"
            smash_cfg["factorizer"] = "qkv_diffusers"
            smash_cfg["cacher"] = "fora"
            smash_cfg["fora_interval"] = 2
            smash_cfg["fora_start_step"] = 2
            smash_cfg["torch_compile_make_portable"] = True
            smash_cfg._prepare_saving = True

            self.model = smash(base_model, smash_cfg)
            print("✅ Optimised model saved to cache")
        else:
            save_the_model = False
            print("📦 Loading optimised model from cache …")
            self.model = PrunaModel.from_pretrained(
                f"/model_cache/{model_cache_name}/"
            )
            print("✅ Optimised model loaded")

        self.model.to("cuda")
        # Warm-up pass (first call after load/compile)
        t0 = time.time()
        self.model(
            image=sample_image, prompt=sample_prompt, guidance_scale=2.5
        )
        warmup_latency = time.time() - t0
        print(f"⏱️  Warm-up latency           : {warmup_latency:.2f}s")

        # Timed optimised inference
        t0 = time.time()
        self.model(
            image=sample_image, prompt=sample_prompt, guidance_scale=2.5
        )
        opt_latency = time.time() - t0
        print(f"⏱️  Optimised inference latency: {opt_latency:.2f}s")

        if save_the_model:
            self.model.save_pretrained(f"/model_cache/{model_cache_name}/")

        # Free the base model – no need to keep it around
        del base_model
        gc.collect()

    @modal.method()
    def inference(self, prompt: str, input_image_url: str) -> bytes:
        input_image = load_image(
            input_image_url
        )
        print("🎨 generating image...")
        output = self.model(
            image=input_image, prompt=prompt, guidance_scale=2.5
        )
        image = output.images[0]
        print("image generated")

        # Convert to bytes
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


@app.function(
    timeout=600 * 60,
    volumes={  # add Volumes to store serializable compilation artifacts and caches
        "/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
    },
)
def run_and_save_output_image(prompt: str, image_url: str):
    t0 = time.time()
    image_bytes = Model().inference.remote(prompt, image_url)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    t0 = time.time()
    image_bytes = Model().inference.remote(prompt, image_url)
    print(f"🎨 Second inference latency: {time.time() - t0:.2f} seconds")

    # Save to a persistent volume
    output_path = Path("/outputs") / f"{int(time.time())}_image.jpg"
    print(f"🎨 Saving output to {output_path}")
    output_path.write_bytes(image_bytes)

    # Also save to tmp folder
    tmp_path = Path("/tmp") / "sd" / "output.jpg"
    tmp_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path.write_bytes(image_bytes)
    print(f"🎨 Also saved output to {tmp_path}")



@app.function(
    timeout=900,  # 15 minutes should be enough for upload
    volumes={
        "/model_cache": modal.Volume.from_name("model_cache", create_if_missing=True),
    },
)
def upload_to_hf(repo_id: str, hf_token: str):
    """Upload a model from cache to Hugging Face.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'username/model-name')
        hf_token: Your Hugging Face API token
    """
    # Upload the model
    model_path = f"/model_cache/{model_cache_name}"
    print(f"Uploading model from {model_path} to {repo_id}")

    # Setup HF API
    api = HfApi(token=hf_token)

    # Create or reuse repo
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)

    # Upload entire folder
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=f"Upload optimized {model_id} with Pruna Pro"
    )

    print(f"✅ Successfully uploaded model to https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def main(
    prompt: str = "Add a hat to the cat",
    image_url: str = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
    upload: bool = False,
    repo_id: str = "",
):

    run_and_save_output_image.remote(prompt=prompt, image_url=image_url)
    if upload:
        if repo_id == "":
            raise ValueError("Provide a value for repo_id")
        hf_token = "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"

        print(f"Uploading model to {repo_id}")
        result = upload_to_hf.remote(repo_id, hf_token)
        print(f"Upload result: {result}")