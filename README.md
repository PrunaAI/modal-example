# Modal AI Model Examples

Run generative AI models on [Modal's serverless platform](https://modal.com).

## Contents

This repository contains examples for running various AI models on Modal. Each model has its own directory with
tutorials and scripts.
There are **two main folders**: Pruna Open Source, and Pruna Pro. For Pruna Pro, you will need a Pruna token,
that you can get [on the Pruna website pricing page](https://www.pruna.ai/pricing)

Within each folder, there will be one folder per model:
```text
Pruna Open Source/
│
├── model-1/
│   ├── smash-model-1-oss.py
│   ├── smash-model-1-tutorial.md
│
Pruna Pro/
│
├── model-2/
│   ├── smash-model-2.py
│   ├── smash-model-2-tutorial.md
```
There may be modal notebooks in the model folder.

These examples demonstrate how to run various generative AI models from different providers on Modal's serverless GPU
platform. By following these examples, you can:

- Deploy powerful image generation models without managing your own GPU infrastructure
- Optimize inference speed using Pruna's acceleration techniques
- Understand how to structure Modal applications for efficient GPU utilization
- Compare performance between base models and optimized versions

## Requirements

- A free [Modal](https://modal.com) account (For more information on getting started with Modal, check out
  the [Modal Getting Started Guide](https://modal.com/docs/guide).)
- A personal [Hugging Face](https://huggingface.co) access token with permission to pull models (for examples using
  Hugging Face models)
- Python 3.9+ and pip

## Setting Up

You may either read the markdown tutorial, or start from the equivalent python script.

To run the python script:

- If applicable, replace the credentials at the top of the file with your credentials (e.g. Huggingface token)
- Run with modal: `modal run <script name>`. Note that modal will automatically kill the job if your connection
  is interrupted. If this is not what you want, you can set it to run in "detach" mode with -d:
  `modal run -d smash-flux-kontext-tutorial.py`
- Beware this scripts are set to run on H100, make sure you are aware of their pricing. Modal includes $30/month of
  free compute in their Starter Plan, which covers ~7 hours of H100 time per month. Still, it may be a good idea to
  develop on smaller machines, keeping in mind that your original model needs to fit on the GPU to be optimized,
  and that the speedup will depend on the hardware (e.g. if you get 2x speedup on hardware A, you may get more, or less,
  on hardware B, so always do your benchmarks in the production conditions)

**Credentials**

You may need a hugging face token or a pruna pro token depending on the model or the optimisation techniques that you
use. In the python files you will find them at the top: `HUGGINGFACE_TOKEN =  "<YOUR HUGGINGFACE TOKEN>"`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
