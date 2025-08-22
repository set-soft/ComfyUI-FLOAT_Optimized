<div align="center">

# ComfyUI FLOAT Optimized

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.09013-b31b1b.svg)](https://arxiv.org/abs/2412.01064)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

**Table of Contents**
1. &#x0001F680; [Installation](#-installation)
2. &#x2600;&#xFE0F; [Usage](#&#xFE0F;-usage)
   - [More examples](#more-examples)
3. &#128190; [Manual Models Download](#-manual-models-download)
   - [Simple](#simple)
   - [Flexible](#flexible)
   - [Very Advanced](#very-advanced)
4. &#128218; [Nodes](#-nodes)
   - [Load FLOAT Models (Opt)](#load-float-models-opt)
   - [FLOAT Process (Opt)](#float-process-opt)
   - [FLOAT Advanced Options](#float-advanced-options)
5. &#128030; [Debugging](#-debugging)
6. &#x0001F4DC; [Project History](#-project-history)
7. &#128279; [Citation](#-citation-of-the-paper)
8. &#128101; [Attributions](#-attributions)
9. &COPY;&#xFE0F; [License](#&#xFE0F;-license)
---

This project provides a ComfyUI wrapper of [FLOAT](https://github.com/deepbrainai-research/float) for Generative Motion Latent Flow Matching for Audio-driven Talking Portrait

The code was optimized to reduce VRAM usage and avoid temporal files.

If you don't know about FLOAT, want to understand more, and you don't want to read the paper, the following explanation might help you [Understanding FLOAT](doc/understanding_FLOAT.md)

> [!WARNING]
> **FLOAT is not for commercial use.**
> Please refer to the licensing terms for more details.


[![FLOAT Encoder Architecture Diagram](doc/nodes.jpg)](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_workflow.json)


## &#x0001F680; Installation

```bash
git clone https://github.com/set-soft/ComfyUI-FLOAT_Optimized.git
cd ./ComfyUI-FLOAT_Optimized
pip install -r requirements.txt
```

Note:
- The code uses `torch` which is installed for ComfyUI, is part of its dependencies.
  This dependency isn't listed to avoid messing with it, which delicated.
- I tested the nodes using an RTX3060 with 12 GB of VRAM and 32 GB of RAM, in addition I have 32 GB of swap (virtual RAM).

## &#x2600;&#xFE0F; Usage

- Load [example workflow](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_workflow.json)<br>
  Or you can use a version of the workflow that can download the example image and audio:
  [quick example workflow](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_workflow_quick.json)
  Note that this version needs some extra nodes.
- Upload driving image and audio, click queue.
  You can get the image originally used from [here](https://raw.githubusercontent.com/deepbrainai-research/float/refs/heads/main/assets/sam_altman_512x512.jpg),
  and the audio from [here](https://github.com/deepbrainai-research/float/raw/refs/heads/main/assets/aud-sample-vs-1.wav)
- Models are automatically downloaded to `/ComfyUI/models/float`.
  But you can also download them manually.

> [!IMPORTANT]
> If models are automatically downloaded you'll see the workflow stopped at the "Load Float Models (Opt)" for a while.<br>
> The download progress will be displayed in the node and also in the console.<br>
> It will have to download 2.4 GB

### More examples

In addition to the [main example](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_workflow.json)
you can also try:

- [Quick](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_workflow_quick.json): A basic example that can download an image and audio example. The downloaded media is from the original paper.
- [Advanced](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_adv.json): This example uses the *advanced* nodes. They show the architecture better. But I recommend using the *very advanced* nodes.
- [Advanced (without comments)](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_adv_no_comments.json): This is the same as *Advanced* but without the notes.
- [Very Advanced](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_very_advanced.json): This example uses the *very advanced* nodes. You can play with the inner details using it. I should also use less VRAM.
- [Very Advanced (without comments)](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_very_advanced_no_comments.json): Same as *Very Advanced* but without the notes.
- [Very Advanced Subgraphs](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_very_advanced_subgraphs.json): This is a *very advanced* version that uses subgraphs, so it can be seen as simple as the basic workflow, but you can go one level down and see it as the *advanced* version, and you can go deep to the *very advanced* level.
- [Very Advanced Dynamic Emotion](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_va_dynamic_emo.json): An experimental workflow that uses the emotion as it changes during the audio clip, not just the emotion for the overall audio clip.
- [Very Advanced Insert](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_very_advanced_sub_insert.json): Very advanced version that reinserts the generated animation in the original image, uses subgraphs
- [Very Advanced Insert 2](https://github.com/set-soft/ComfyUI-FLOAT_Optimized/raw/refs/heads/master/example_workflows/float_very_advanced_sub_insert_2.json): Same as *Very Advanced Insert* but shows how to handle two faces.


### &#128190; Manual models download

Models are automatically downloaded, but you can also download them manually.
This is for advanced use, not usually needed.
There are two ways to do it.

#### Simple

Just download the unified [FLOAT model](https://huggingface.co/set-soft/float/resolve/main/FLOAT.safetensors?download=true)
to a folder named `models/float` inside your ComfyUI installation.

This file (2.4 GiB) contains the weights for all the networks used by FLOAT.

#### Flexible

Three *models* are needed.

1. Wav2Vec 2.0

    This is an audio encoder used as base for speech recognition. Was created by FaceBook AI.
    You can download the files to a folder named `models/audio/wav2vec2-base-960h` inside your ComfyUI installation.
    Note that you don't need to include *pytorch_model.bin* or *tf_model.h5*, you just need the JSON files and *model.safetensors*
    - Repo: [HuggingFace repo](https://huggingface.co/facebook/wav2vec2-base-960h).
    - License: [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
    - Paper: https://huggingface.co/papers/2006.11477

2. Speech Emotion Recognition

    This is what FLOAT uses to detect the emotion in the audio, uses Wav2Vec 2.0 as base.
    Well, in fact is based on another net that uses Wav2Vec ([Base](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english))
    You can download the files to a folder named `models/audio/wav2vec-english-speech-emotion-recognition` inside your ComfyUI installation.
    - Repo: [HuggingFace repo](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition).
    - License: [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
    - Paper: doi 10.57967/hf/3569 (for the base speech recognition model)

3. FLOAT

    This is the main model.
    You can download the file to a folder named `models/float` inside your ComfyUI installation.
    - Repo: [GitHub page](https://github.com/deepbrainai-research/float)
            [download 1](https://drive.google.com/file/d/1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0/view?pli=1)
            [download 2](https://huggingface.co/yuvraj108c/float/resolve/main/float.pth?download=true)
    - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

#### Very Advanced

The current **VA** (Very Advanced) nodes can either download the components or extract them from
the unified 2.4 GB file.

But if you want to manually download them what the nodes needs is:

1. The Wav2Vec 2.0 and Speech Emotion Recognition models described before (models/audio)
2. The FLOAT individual parts found [here](https://huggingface.co/set-soft/float_advanced/tree/main)
   Download the folders and copy them to `models/float` inside your ComfyUI installation.

These nodes are the most flexible and they enable using various options to fine-tune the FLOAT behavior.
Note that their input and outputs might change.

## &#128218; Nodes

The nodes come in three flavors:

- Regular nodes: simple to use but no flexibility and needs more memory. Recommended for casual use.
- [Advanced nodes](doc/nodes_adv.md): uses the same model files as the regular nodes
- [Very advanced](doc/nodes_vadv.md): uses separated portions and provides flexibility. Recommended for investigating options.

### Load FLOAT Models (Opt)

- **model**: Shows the models in `models/float`. Currently you can choose between `FLOAT.safetensors` (unified) and `float.pth`
  The default is `FLOAT.safetensors`, if the file isn't there it will be downloaded.
- **target_device**: Selects the inference device. Might be useful if you have more than one GPU.
- **cudnn_benchmark**: When enabled CUDA will try to find the best algorithm to run the inference steps.
  The drawback is that this makes the first inference run very slow.
  For this reason the default value is disabled. This is much better for RTX3060 systems.
  If you find enabling it is better for your system please report it.

### FLOAT Process (Opt)

- **ref_image**: Image to apply the voice. Use a square image. The net was trained using 512x512 images,
  so your image will be rescaled to this size. Use simple backgrounds for better results. Leave enough
  space around the face to allow for head motion or just enable **face_align**.
- **ref_audio**: The voice to use. If this is a song try removing the music. The model can detect emotions,
  but it was trained for english. The length of the generated video is the same of the audio. Longer audios
  will need more memory.
- **float_pipe**: Connect the `Load Float Models (Opt)` node here.
- **fps**: Frames Per Second, 25 fps is fine, 30 will probably sync better with your screen. Higher FPSs
  will need more memory.
- **emotion**: Can be used to shift the emotion of the reference image.
- **face_align**: When enabled the image will be processed to detect the face and ensure the space around
  it is suitable for head motion. If disabled you must ensure it.
- **seed**: random seed for the generation, change it to get different videos.
- **control after generate**: added by ComfyUI to choose what to do after a generation. Use *fixed* to
  keep the same **seed**, allowing repetitibility.

### FLOAT Advanced Options

- **r_cfg_scale**: Reference classifier-free guidance (vector field) scale.
  Will just enable CFG process if different than 1.
- **attention_window**: Attention window size, e.g., if 1, attend frames of t-1, t, t+1 for frame t
- **audio_dropout_prob**: Dropout probability for audio
- **ref_dropout_prob**: Dropout probability for reference
- **emotion_dropout_prob**: Dropout probability for emotion
- **ode_atol**: Absolute tolerance for the Ordinary Differential Equation solver (ODE)
- **ode_rtol**: Relative tolerance for the ODE
- **nfe**: Number of Function Evaluations for the ODE
- **torchdiffeq_ode_method**: ODE method
- **face_margin**: Controls the space around the face. The network was trained using 1.6. Making it bigger
  you'll get more margin. Best results are achieved using 1.6, but sometimes this produces artifacts with
  the hair, you can try to enlarge or reduce the margin a little.
- **rgba_conversion**: How to handle images with alpha channel. Three strategies:
    1. **blend_with_color** will blend the image with the specified color
    2. **discard_alpha** the alpha channel is just ignored
    3. **replace_with_color** fully transparent pixels are replaced by the specified color
    Nodes like *Inspyrenet Rembg* generate RGBA images, part of
    [ComfyUI-Inspyrenet-Rembg](https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg)
- **bkg_color_hex**: Color used for the *rgba_conversion*. You can connect a *LayerUtility: ColorPicker* node
  here, part of [ComfyUI-LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle).


## &#128030; Debugging

When you face problems you can ask these nodes to show more information.

- You can just run ComfyUI using `--verbose DEBUG`.
  This will show extra information for *all* the ComfyUI operations
- If you just want extra information for these nodes you can define the `FLOAT_OPTIMIZED_NODES_DEBUG` environment variable to `1`.
  This will show extra information related to FLOAT nodes.
- If you want even more information use `2` or `3` for the environment variable.


## &#x0001F4DC; Project History

- 1.0.0 2025-06-14: Initial release
  - Optimized
  - Advanced nodes
  - Very advanced nodes

- 1.1.0 2025-07-03: Dynamic Emotion and fixes
  - Fixed support for broken Transformers (default attn_implementation is incompatible)
  - Fixed support for Apple Silicon Macs (#2)
  - Added dynamic emotions handling (experimental)


## &#128279; Citation of the paper

```bibtex
@article{ki2024float,
  title={FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait},
  author={Ki, Taekyung and Min, Dongchan and Chae, Gyeongsu},
  journal={arXiv preprint arXiv:2412.01064},
  year={2024}
}
```

## &#128101; Attributions

- **FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait** by
  [Taekyung Ki](https://taekyungki.github.io), [Dongchan Min](https://kevinmin95.github.io), [Gyeongsu Chae](https://www.aistudios.com/ko)
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Wav2Vec 2.0**  by
  Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli from [FaceBook AI](https://ai.meta.com/)
  [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
- **Speech Emotion Recognition By Fine-Tuning Wav2Vec 2.0** by
  [Rob Field](https://huggingface.co/r-f) et al.
  [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
- **Base FLOAT nodes for ComfyUI** by Yuvraj Seegolam [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Optimizations** by Salvador E. Tropea [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## &COPY;&#65039; License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
