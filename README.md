# Thai-TTS-Evaluation
This repository hosts a Thai Text-to-Speech (TTS) evaluation script, focusing on assessing speaker tone and pronunciation performance.

## Speaker Encoder Model
For the speaker tone objective, we utilized the Speaker Encoder Cosine Similarity (SECS) metric to assess the resemblance between the synthesized speech and the original speaker's speech. This method involves calculating the cosine similarity between the speaker embeddings derived from two speech samples, using a speaker encoder. We utilize the [Coqui speaker encoder](https://github.com/coqui-ai/TTS/releases/tag/speaker\_encoder\_model), trained on the comprehensive VoxCeleb dataset, ensuring broad generalizability in our evaluations.

## Speech-to-TExt Model
For pronunciation, we utilized a [Thai speech-to-text model](https://huggingface.co/biodatlab/whisper-th-medium-combined). The underlying assumption is that high-quality synthesized speech should yield similar speech-to-text results as the original speech. This method allows us to gauge the accuracy of pronunciation in the synthesized speech.
