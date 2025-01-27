# Whisper Model Card

## Overview
Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation, proposed in the paper *Robust Speech Recognition via Large-Scale Weak Supervision* by Alec Radford et al. from OpenAI. Trained on over 5 million hours of labeled data, Whisper demonstrates a strong ability to generalize to many datasets and domains in a zero-shot setting.

## Model Variants
Whisper large-v3-turbo is a finetuned version of a pruned Whisper large-v3. It has reduced the number of decoding layers from 32 to 4, making the model faster with minor quality degradation.

## Disclaimer
Content for this model card has partly been written by the 🤗 Hugging Face team and partly copied and pasted from the original model card.

## Usage
Whisper large-v3-turbo is supported in Hugging Face 🤗 Transformers. To run the model, first install the necessary libraries:

```bash
pip install --upgrade pip
pip install --upgrade transformers datasets[audio] accelerate
```

### Transcribing Audio
The model can be used with the pipeline class to transcribe audios of arbitrary length:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

### Transcribing Local Audio Files
To transcribe a local audio file, pass the path to your audio file when you call the pipeline:

```python
result = pipe("audio.mp3")
```

### Batch Transcription
Multiple audio files can be transcribed in parallel by specifying them as a list and setting the `batch_size` parameter:

```python
result = pipe(["audio_1.mp3", "audio_2.mp3"], batch_size=2)
```

### Decoding Strategies
Transformers is compatible with all Whisper decoding strategies, such as temperature fallback and condition on previous tokens. Example:

```python
generate_kwargs = {
    "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

result = pipe(sample, generate_kwargs=generate_kwargs)
```

### Language and Task Specification
Whisper predicts the language of the source audio automatically. If the source audio language is known a-priori, it can be passed as an argument to the pipeline:

```python
result = pipe(sample, generate_kwargs={"language": "english"})
```

To perform speech translation, set the task to "translate":

```python
result = pipe(sample, generate_kwargs={"task": "translate"})
```

### Timestamps
For sentence-level timestamps:

```python
result = pipe(sample, return_timestamps=True)
print(result["chunks"])
```

For word-level timestamps:

```python
result = pipe(sample, return_timestamps="word")
print(result["chunks"])
```

## Additional Speed & Memory Improvements
### Chunked Long-Form
Whisper has a receptive field of 30-seconds. To transcribe audios longer than this, use one of two long-form algorithms: Sequential or Chunked.

### Torch Compile
The Whisper forward pass is compatible with `torch.compile` for speed-ups:

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

for _ in tqdm(range(2), desc="Warm-up step"):
    with sdpa_kernel(SDPBackend.MATH):
        result = pipe(sample.copy(), generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})

with sdpa_kernel(SDPBackend.MATH):
    result = pipe(sample.copy())

print(result["text"])
```

### Flash Attention 2
Install Flash Attention:

```bash
pip install flash-attn --no-build-isolation
```

Then pass `attn_implementation="flash_attention_2"` to `from_pretrained`:

```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
```

### Torch Scale-Product-Attention (SDPA)
If your GPU does not support Flash Attention, use PyTorch scaled dot-product attention (SDPA):

```python
from transformers.utils import is_torch_sdpa_available

print(is_torch_sdpa_available())
```

If True, SDPA is activated by default. Otherwise, upgrade your PyTorch version.

## Model Details
Whisper is a Transformer-based encoder-decoder model. There are two flavors: English-only and multilingual. The model predicts transcriptions in the same language as the audio for speech recognition and in a different language for speech translation.

### Checkpoints
Whisper checkpoints come in five configurations:

| Size       | Parameters | English-only | Multilingual |
|------------|-------------|--------------|--------------|
| tiny       | 39 M        | ✓            | ✓            |
| base       | 74 M        | ✓            | ✓            |
| small      | 244 M       | ✓            | ✓            |
| medium     | 769 M       | ✓            | ✓            |
| large      | 1550 M      | x            | ✓            |
| large-v2   | 1550 M      | x            | ✓            |
| large-v3   | 1550 M      | x            | ✓            |
| large-v3-turbo | 809 M   | x            | ✓            |

## Fine-Tuning
The pre-trained Whisper model can be fine-tuned for specific languages and tasks. Refer to the blog post *Fine-Tune Whisper with 🤗 Transformers* for a step-by-step guide.

## Evaluated Use
Whisper is intended for AI researchers and developers. It is useful for ASR and speech translation tasks but should be evaluated in specific contexts before deployment.

## Performance and Limitations
Whisper models show improved robustness to accents, background noise, and technical language. However, they may include hallucinations and exhibit uneven performance across languages and accents.

## Broader Implications
Whisper models can improve accessibility tools but may also enable surveillance technologies. Users sh
