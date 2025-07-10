import time
from io import BytesIO
from pathlib import Path

import modal

HUGGINGFACE_TOKEN =  "<YOUR HUGGINGFACE TOKEN>"
PRUNA_TOKEN = "<YOUR PRUNA PRO TOKEN>"

tag = "12.6.0-devel-ubuntu22.04"
cuda_image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")

cuda_image = cuda_image.pip_install(
    "pruna-pro",
    "diffusers",
    "pillow",
    pre=True,
)

app = modal.App("flux-dev-pruna-pro", image=cuda_image)

with cuda_image.imports():
    from diffusers import FluxPipeline
    from pruna_pro import SmashConfig
    from pruna_pro import smash
    from pruna_pro import PrunaProModel
    import torch
    import gc


MINUTES = 60  # seconds
MODEL_ID = f"black-forest-labs/FLUX.1-dev"


@app.cls(
    gpu="H100",  # fastest GPU on Modal
    scaledown_window=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts and caches
        "/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
)
class Model:
    @modal.enter()
    def enter(self):
        """
        Load the base model, run a one-off benchmark, then generate inference.
        """
        # ---------- BASE model -------------------------------------------
        sample_prompt = "A playful cat wearing a blue wizard hat"

        # ---------- BASE model -------------------------------------------
        print("🔄 Loading base model …")
        t0 = time.time()
        base_model = FluxPipeline.from_pretrained(
            MODEL_ID, cache_dir="/cache",
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16,
        ).to("cuda")
        base_load_time = time.time() - t0
        print(f"✅ Base model loaded in {base_load_time:.2f}s")

        # Inference latency with base model
        t0 = time.time()
        base_model(
            prompt=sample_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        base_latency = time.time() - t0
        print(f"⏱️  Base inference latency: {base_latency:.2f}s")

        # ---------- OPTIMISED model --------------------------------------
        print("🚀 Optimising model with Pruna Smash …")
        smash_config = SmashConfig()
        smash_config["cacher"] = "auto"
        smash_config["auto_cache_mode"] = "taylor"
        smash_config["auto_speed_factor"] = 0.4
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False

        self.model = smash(base_model, smash_config, token=PRUNA_TOKEN)

        self.model.to("cuda")
        # Warm-up pass (first call after load/compile)
        t0 = time.time()
        self.model(
            prompt=sample_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        )
        warmup_latency = time.time() - t0
        print(f"⏱️Warm-up latency: {warmup_latency:.2f}s")

        # Timed optimised inference
        t0 = time.time()
        self.model(
            prompt=sample_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        )
        opt_latency = time.time() - t0
        print(f"⏱️Optimised inference latency: {opt_latency:.2f}s")

        # Free the base model – no need to keep it around
        del base_model
        gc.collect()

    @modal.method()
    def inference(self, prompt: str, num_inference_steps=30, guidance_scale=7.5) -> bytes:
        print("🎨 generating image...")
        output = self.model(prompt=prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            )
        image = output.images[0]
        print("image generated")

        # Convert to bytes
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()

@app.function(
timeout=60 * MINUTES,
volumes={
    "/outputs": modal.Volume.from_name("outputs", create_if_missing=True)}
)
def run_and_store_inference(prompt: str, num_inference_steps=30, guidance_scale=7.5):
    image_bytes = Model().inference.remote(prompt,
                                           num_inference_steps=num_inference_steps,
                                           guidance_scale=guidance_scale)
    print("🎨 first inference done, model is warm.")

    t0 = time.time()
    image_bytes = Model().inference.remote(prompt,
                                           num_inference_steps=num_inference_steps,
                                           guidance_scale=guidance_scale)
    print(f"🎨 Second inference latency: {time.time() - t0:.2f} seconds")

    # Save to a persistent volume
    output_path = Path("/outputs") / f"{int(time.time())}_image.jpg"
    print(f"🎨 Saving output to {output_path}")
    output_path.write_bytes(image_bytes)


@app.local_entrypoint()
def main(
        prompt: str = "Add a hat to the cat",
):
    run_and_store_inference.remote(prompt=prompt)
