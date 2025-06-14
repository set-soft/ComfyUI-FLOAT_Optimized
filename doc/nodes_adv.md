# FLOAT (Advanced) Nodes Reference

This document provides a reference for the "Advanced" (Ad) nodes in the ComfyUI FLOAT Optimized integration. These nodes provide a middle ground of control, allowing users to adjust key parts of the pipeline like face alignment and sampling parameters while still using the convenient, all-in-one `float_pipe`.

**Jump to section:**
- [FloatImageFaceAlign](#floatimagefacealign)
- [FloatAdvancedParameters](#floatadvancedparameters)
- [FloatEncodeImageToLatents](#floatencodeimagetolatents)
- [FloatGetIdentityReference](#floatgetidentityreference)
- [FloatEncodeAudioToLatentWA](#floatencodeaudiotolatentwa)
- [FloatEncodeEmotionToLatentWE](#floatencodeemotiontolatentwe)
- [FloatSampleMotionSequenceRD](#floatsamplemotionsequencerd)
- [FloatDecodeLatentsToImages](#floatdeodelatentstoimages)

---

### `FloatImageFaceAlign`
- **Display Name:** Face Align for FLOAT (Ad)
- **Description:** Pre-processes an image for the FLOAT pipeline. It performs face detection, crops the image around the face with a specified margin, resizes it to a target resolution, and handles the conversion of RGBA images to RGB.
- **Inputs:**
  - `image`: (IMAGE) The input image or image batch to process.
  - `face_margin`: (FLOAT) The multiplier for the detected face bounding box to determine the final crop size. Larger values mean a wider shot.
  - `rgba_conversion`: (Dropdown) The strategy to use when converting a 4-channel RGBA image to a 3-channel RGB image.
  - `bkg_color_hex`: (STRING) The background color (in hex format, e.g., `#000000`) to use for blending or replacement during RGBA conversion.
  - `size`: (INT, *Optional*) The target output size (height and width). If not connected, it uses the default size from the model's options (usually 512). Can be linked from a loader node's `inferred_input_size`.
- **Outputs:**
  - `image`: (IMAGE) The processed image batch, cropped, resized, and ready for the FLOAT Encoder.

### `FloatAdvancedParameters`
- **Display Name:** FLOAT Advanced Options (Ad)
- **Description:** A utility node that groups together numerous advanced hyperparameters into a single dictionary (`ADV_FLOAT_DICT`). This dictionary can be passed to loader nodes to configure the model pipeline with non-default settings.
- **Inputs:**
  - **`r_cfg_scale`** (FLOAT): **Reference Identity Guidance Scale.** Controls adherence to the identity when using the experimental `include_r_cfg` feature in the sampler.
  - **`attention_window`** (INT): **Attention Window Size.** An architectural hyperparameter for the `FlowMatchingTransformer`, defining the local window size for the attention mask. Must match the value the FMT model was trained with.
  - **`audio_dropout_prob`** (FLOAT): The probability of nulling out the audio condition (`wa`) during ODE sampling. Used for classifier-free guidance and can add variation.
  - **`ref_dropout_prob`** (FLOAT): The probability of nulling out the reference identity condition (`wr`) during ODE sampling.
  - **`emotion_dropout_prob`** (FLOAT): The probability of nulling out the emotion condition (`we`) during ODE sampling.
  - **`ode_atol`** (FLOAT): **Absolute Tolerance** for the ODE solver. A smaller value increases precision but can slow down computation.
  - **`ode_rtol`** (FLOAT): **Relative Tolerance** for the ODE solver. A smaller value increases precision but can slow down computation.
  - **`nfe`** (INT): **Number of Function Evaluations.** The number of steps for the ODE solver. Higher values can increase quality and detail at the cost of longer generation time.
  - **`torchdiffeq_ode_method`** (Dropdown): The specific fixed-step numerical integration method for the ODE solver (e.g., 'euler', 'midpoint', 'rk4').
  - **`face_margin`** (FLOAT): The default multiplier for the detected face bounding box used by face alignment nodes.
  - **`rgba_conversion`** (Dropdown): The default strategy to use when converting a 4-channel RGBA image to a 3-channel RGB image.
  - **`bkg_color_hex`** (STRING): The default background color (in hex format, e.g., `#000000`) to use for blending or replacement during RGBA conversion.
- **Outputs:**
  - `advanced_options`: (ADV_FLOAT_DICT) A dictionary containing all the specified hyperparameter values.

### `FloatEncodeImageToLatents`
- **Display Name:** FLOAT Encode Image to Latents (Ad)
- **Description:** Encodes a batch of reference images into the core latents required by FLOAT, using the `motion_autoencoder` from the main `float_pipe`.
- **Inputs:**
  - `ref_image`: (IMAGE) A batch of reference images, correctly sized for the encoder.
  - `float_pipe`: (FLOAT_PIPE) The main loaded FLOAT pipeline.
- **Outputs:**
  - `appearance_pipe (Wsr)`: (FLOAT_APPEARANCE_PIPE) A pipe containing the appearance latent (`h_source` / `s_r`) and the list of feature maps (`feats`).
  - `r_s_lambda_latent`: (TORCH_TENSOR) The motion control parameters (`h_motion`) derived from the reference image.
  - `float_pipe`: (FLOAT_PIPE) Passthrough of the input pipe.

### `FloatGetIdentityReference`
- **Display Name:** FLOAT Get Identity Reference (Ad)
- **Description:** Derives the batched identity reference latent (`r_s` or `Wrs`) from the `r_s_lambda_latent`. It uses the `direction` module within the `float_pipe`'s decoder to perform the transformation.
- **Inputs:**
  - `r_s_lambda_latent`: (TORCH_TENSOR) The motion control parameters from the `FloatEncodeImageToLatents` node.
  - `float_pipe`: (FLOAT_PIPE) The main loaded FLOAT pipeline.
- **Outputs:**
  - `r_s_latent`: (TORCH_TENSOR) The final reference identity latent.
  - `float_pipe`: (FLOAT_PIPE) Passthrough of the input pipe.

### `FloatEncodeAudioToLatentWA`
- **Display Name:** FLOAT Encode Audio to latent wa (Ad)
- **Description:** Takes raw audio, resamples and preprocesses it using the `float_pipe`'s internal data processor, and then encodes it into the audio conditioning sequence (`wa_latent`).
- **Inputs:**
  - `float_pipe`: (FLOAT_PIPE) The main loaded FLOAT pipeline.
  - `audio`: (AUDIO) The raw ComfyUI audio input.
  - `fps`: (FLOAT) The target video frames-per-second, used to determine the length of the output latent sequence.
- **Outputs:**
  - `wa_latent`: (TORCH_TENSOR) The final audio conditioning latent sequence.
  - `audio_num_frames`: (INT) The total number of frames calculated from the audio length and FPS.
  - `preprocessed_audio`: (TORCH_TENSOR) The audio after feature extraction (but before the main Wav2Vec model), ready for the emotion encoder.
  - `float_pipe`: (FLOAT_PIPE) Passthrough of the input pipe.

### `FloatEncodeEmotionToLatentWE`
- **Display Name:** FLOAT Encode Emotion to latent we (Ad)
- **Description:** Generates the emotion conditioning latent (`we`). It uses the `float_pipe`'s internal emotion encoder to either predict the emotion from preprocessed audio features or create a one-hot encoding for a user-specified emotion.
- **Inputs:**
  - `preprocessed_audio`: (TORCH_TENSOR) The preprocessed audio features from the `FloatEncodeAudioToLatentWA` node.
  - `float_pipe`: (FLOAT_PIPE) The main loaded FLOAT pipeline.
  - `emotion`: (Dropdown) Select a specific emotion or 'none' to have the model predict from the audio.
- **Outputs:**
  - `we_latent`: (TORCH_TENSOR) The final emotion conditioning latent.
  - `float_pipe`: (FLOAT_PIPE) Passthrough of the input pipe.

### `FloatSampleMotionSequenceRD`
- **Display Name:** FLOAT Sample Motion Sequence rd (Ad)
- **Description:** The core sampling node. It uses the Flow Matching Transformer (FMT) from the `float_pipe` and an ODE solver to generate the driven motion latent sequence (`r_d`). It uses ODE and some CFG parameters configured within the `float_pipe`.
- **Inputs:**
  - `r_s_latent`: (TORCH_TENSOR) The reference identity latent (`wr`).
  - `wa_latent`: (TORCH_TENSOR) The audio conditioning latent (`wa`).
  - `audio_num_frames`: (INT, *Link Only*) The total number of frames to generate.
  - `we_latent`: (TORCH_TENSOR) The emotion conditioning latent (`we`).
  - `float_pipe`: (FLOAT_PIPE) The main loaded FLOAT pipeline, which provides the FMT model and ODE/CFG settings.
  - `a_cfg_scale`: (FLOAT) Audio Guidance Scale.
  - `e_cfg_scale`: (FLOAT) Emotion Guidance Scale.
  - `seed`: (INT) The seed for the random noise generator.
- **Outputs:**
  - `r_d_latents`: (TORCH_TENSOR) The generated sequence of driven motion latents (`WrD`).
  - `float_pipe`: (FLOAT_PIPE) Passthrough of the input pipe.

### `FloatDecodeLatentsToImages`
- **Display Name:** FLOAT Decode Latents to Images (Ad)
- **Description:** The final image generation step. It uses the decoder from the `float_pipe`'s `motion_autoencoder` to render the final animated image sequence from the appearance latents (`s_r`, `s_r_feats`) and the driven motion sequence (`r_d`).
- **Inputs:**
  - `appearance_pipe (Ws→r)`: (FLOAT_APPEARANCE_PIPE) The bundled appearance information from `FloatEncodeImageToLatents`.
  - `r_d_latents`: (TORCH_TENSOR) The driven motion latent sequence from the sampler.
  - `float_pipe`: (FLOAT_PIPE) The main loaded FLOAT pipeline.
- **Outputs:**
  - `images`: (IMAGE) The final generated image sequence, concatenated into a single batch.
  - `fps`: (FLOAT) The frames-per-second value used for the generation, passed through for video assembly nodes.
  - `float_pipe`: (FLOAT_PIPE) Passthrough of the input pipe.
