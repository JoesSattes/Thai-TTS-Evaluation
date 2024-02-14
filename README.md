# Thai-TTS-Evaluation
This repository hosts a Thai Text-to-Speech (TTS) evaluation script, focusing on assessing speaker tone and pronunciation performance.

## Speaker Encoder Model
For the speaker tone objective, we utilized the Speaker Encoder Cosine Similarity (SECS) metric to assess the resemblance between the synthesized speech and the original speaker's speech. This method involves calculating the cosine similarity between the speaker embeddings derived from two speech samples, using a speaker encoder. We utilize the [Coqui speaker encoder](https://github.com/coqui-ai/TTS/releases/tag/speaker\_encoder\_model), trained on the comprehensive VoxCeleb1, VoxCeleb2, and all language CommonVoice datasets, ensuring broad generalizability in our evaluations.

## Speech-to-Text Model
For pronunciation, we utilized a [Thai speech-to-text model](https://huggingface.co/biodatlab/whisper-th-medium-combined). The underlying assumption is that high-quality synthesized speech should yield similar speech-to-text results as the original speech. This method allows us to gauge the accuracy of pronunciation in the synthesized speech.

## Requirements
- python==3.7.13
- TTS==0.8.0
- webrtcvad==2.0.10
- torch==1.13.0
- torch-audiomentations==0.11.0
- torch-complex==0.4.3
- torch-pitch-shift==1.2.2
- torchaudio==0.13.0
- torchmetrics==0.8.0
- numpy==1.21.6
- huggingface-hub==0.14.1
- transformers==4.25.1

## Setup

1. **Environment Setup**: Ensure that you have a compatible Python environment. Using Conda or virtualenv is recommended to manage dependencies.
   
   ```bash
   conda create --name voice_env python=3.7.13
   conda activate voice_env
   ```

2. **Install Dependencies**:
   
   ```bash
   pip install <requirement lists>
   ```

3. **GPU Support**: If you're planning to use a GPU, ensure that your PyTorch installation is compatible with your CUDA version.


## Running
```
python evaluate_tts.py
```

## Acknowledgement
Coming soon
