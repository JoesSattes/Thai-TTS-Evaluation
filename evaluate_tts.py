# Required for inline display of matplotlib plots in Jupyter notebooks.
%matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd

# Standard library imports
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Imports from custom modules for handling data and models
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, _clean_text

# Additional imports for audio processing and evaluation
from scipy.io.wavfile import write
import numpy as np 
import pandas as pd
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import string
import time
import argparse

# TTS and audio processing imports
try:
    from TTS.utils.audio import AudioProcessor
except ImportError:
    from TTS.utils.audio import AudioProcessor
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

# Import for Thai language tokenization
from pythainlp.tokenize import word_tokenize

# Configuration setup for the TTS model
def config(path_model):
    """
    Set up the configuration for the TTS model including paths and model settings.
    """
    # Update these path
    MODEL_PATH = os.path.join(path_model, 'checkpoint_230000.pth')
    CONFIG_PATH = os.path.join(path_model, 'config.json')
    TTS_LANGUAGES = os.path.join(path_model, "language_ids.json")
    TTS_SPEAKERS = 'config_se.json'
    SPEAKER_PATH = os.path.join(path_model, 'speakers.pth')
    USE_CUDA = torch.cuda.is_available()
    return MODEL_PATH, CONFIG_PATH, TTS_SPEAKERS, USE_CUDA, TTS_LANGUAGES

# Function to create speaker embeddings
def createSpeakerEmbed(model, path_file: str):
    """
    Generate speaker embeddings from an audio file.
    """
    embed = model.speaker_manager.compute_embedding_from_clip(wav_file=path_file)
    return embed

# Function to calculate cosine similarity
def calculateCosineSim(vector_og, vector_speech_synthesis):
    """
    Calculate the cosine similarity between two vectors.
    """
    cos_sim = cosine_similarity(vector_og, vector_speech_synthesis)
    return cos_sim

# Main function for internal cosine similarity calculation
def internal_calculate_cosine_sim(model, ap, MODEL_PATH, CONFIG_PATH, stt_model):
    """
    Calculate cosine similarity and other metrics for model evaluation.
    """
    from evaluate import load
    wer_metric = load("wer")
    cer_metric = load("cer")
    
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(MODEL_PATH, net_g, None)
    
    # Load and process dataset for evaluation
    dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    loader = DataLoader(dataset, num_workers=8, shuffle=False, batch_size=1, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    # Iterate through the dataset and perform evaluations
    data_list = list(loader)
    list_embed_eval = []
    list_embed_gen = []
    sid_src_list = []
    gen_stt_list, label_stt_list = [], []
    label_t1_stt, label_t2_stt, pred_t1_stt, pred_t2_stt = [], [], [], []
    for d in data_list:
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda() for x in d]
        with torch.no_grad():
            audio = net_g.infer(x, x_lengths, sid=sid_src, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        lab_mel = model.speaker_manager.encoder_ap.melspectrogram(y.cpu().numpy().squeeze())
        lab_mel = torch.from_numpy(lab_mel)
        lab_d_vector = model.speaker_manager.encoder.compute_embedding(lab_mel)
        
        pred_mel = model.speaker_manager.encoder_ap.melspectrogram(audio.squeeze())
        pred_mel = torch.from_numpy(pred_mel)
        pred_d_vector = model.speaker_manager.encoder.compute_embedding(pred_mel)
        
        ipd.display(ipd.Audio(y[0].cpu().numpy(), rate=hps.data.sampling_rate, normalize=False))
        ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

        list_embed_eval.append(lab_d_vector)
        list_embed_gen.append(pred_d_vector)
        
        gen_stt_list.append(" ".join(word_tokenize(stt_model(y[0].cpu().numpy()[0])['text'], engine="newmm")))
        label_stt_list.append(" ".join(word_tokenize(stt_model(audio)['text'], engine="newmm")))
        
        sid_src_list.append(sid_src)

    list_embed_eval = np.array(list_embed_eval)
    list_embed_gen = np.array(list_embed_gen)
    list_score = []
    list_score_t1, list_score_t2 = [], []

    if len(list_embed_eval) == len(list_embed_gen):
        print(f"The number of list_embed_eval equal to number of list_embed_gen:  {len(list_embed_gen)}")
    else:
        print("Error")

    for i in range(len(list_embed_gen)):
        score = calculateCosineSim(list_embed_eval[i].reshape(1, -1), list_embed_gen[i].reshape(1, -1))
        if sid_src_list[i] == 0:
            list_score_t1.append(score)
            label_t1_stt.append(label_stt_list[i])
            pred_t1_stt.append(gen_stt_list[i])
        elif sid_src_list[i] == 1:
            list_score_t2.append(score)
            label_t2_stt.append(label_stt_list[i])
            pred_t2_stt.append(gen_stt_list[i])
        else: assert False, "sid_src is not compatible for Tsync dataset"
        list_score.append(score)  
    
    # Displaying audio and calculating embeddings
    find_avg = np.sum(list_score_t1) / len(list_score_t1)
    find_std = np.std(list_score_t1)
    print(f"Overall T1 Score : {find_avg} +- {find_std}")
    find_avg = np.sum(list_score_t2) / len(list_score_t2)
    find_std = np.std(list_score_t2)
    print(f"Overall T2 Score : {find_avg} +- {find_std}")
    find_avg = np.sum(list_score) / len(list_score)
    find_std = np.std(list_score)
    print(f"Overall Score : {find_avg} +- {find_std}")
    t1_wer = wer_metric.compute(references=label_t1_stt, predictions=pred_t1_stt)
    t1_cer = cer_metric.compute(references=label_t1_stt, predictions=pred_t1_stt)
    print(f"Text T1 score: WER {t1_wer}, CER {t1_cer}")
    t2_wer = wer_metric.compute(references=label_t2_stt, predictions=pred_t2_stt)
    t2_cer = cer_metric.compute(references=label_t2_stt, predictions=pred_t2_stt)
    print(f"Text T2 score: WER {t2_wer}, CER {t2_cer}")
    return find_avg, find_std

# Speech-to-text processing function using a pre-trained model
def stt_process():
    """
    Set up the speech-to-text processing pipeline.
    """
    from transformers import pipeline
    MODEL_NAME = "biodatlab/whisper-th-medium-combined"  # specify the model name
    lang = "th"  # specify the language

    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(task="automatic-speech-recognition", model=MODEL_NAME, chunk_length_s=30, device=device)
    return pipe

# Main processing function
def main_process(G_MODEL_PATH, G_CONFIG_PATH):
    """
    Main function to process the TTS model and perform evaluations.
    """
    # Update this path
    path_model = "/path/to/coqui_speaker_model"

    # Configuration setup
    MODEL_PATH, CONFIG_PATH, TTS_SPEAKERS, USE_CUDA, TTS_LANGUAGES = config(path_model)
    
    # Load configurations and model
    C = load_config(CONFIG_PATH)
    ap = AudioProcessor(**C.audio)

    # Model setup and loading
    model = setup_model(C)
    cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Adjust and load model weights
    model_weights = cp['model'].copy()
    for key in list(model_weights.keys()):
        if "speaker_encoder" in key:
            del model_weights[key]
    model.load_state_dict(model_weights)
    model.eval()

    if USE_CUDA:
        model = model.cuda()

    # logic for TTS processing and evaluation
    use_griffin_lim = False
    internal_calculate_cosine_sim(model, ap, MODEL_PATH = G_MODEL_PATH, CONFIG_PATH = G_CONFIG_PATH, stt_model=stt_process())

# Function to select TTS model and configuration based on experiment type
def tts_path(exp_type='notrim_phone_vits_1000ep'):
    """
    Select the TTS model and configuration path based on the experiment type.
    """
    # Define paths based on experiment type
    # Update these paths according to your directory structure and model configuration
    if exp_type == 'trim_word_vits_1000ep':
        G_MODEL_PATH = "model/vits/trim_all_tsync_2/G_306000.pth"
        G_CONFIG_PATH = "configs/trim_thai_tsync.json"
    elif exp_type == 'notrim_word_vits_1000ep':
        G_MODEL_PATH = "model/vits/all_tsync_2/G_306000.pth"
        G_CONFIG_PATH = "configs/thai_tsync.json"
    elif exp_type == 'trim_phone_vits_1000ep':
        G_MODEL_PATH = "model/vits/phone_trim_all_tsync_2/G_208000.pth"
        G_CONFIG_PATH = "configs/phone_trim_thai_tsync.json"
    elif exp_type == 'notrim_phone_vits_1000ep':
        G_MODEL_PATH = "model/vits/phone_all_tsync_2/G_208000.pth"
        G_CONFIG_PATH = "configs/phone_thai_tsync.json"
    elif exp_type == 'trim_syl_vits_1000ep':
        G_MODEL_PATH = "model/vits/syl_trim_all_tsync_2/G_300000.pth"
        G_CONFIG_PATH = "configs/syl_trim_thai_tsync.json"
    elif exp_type == 'notrim_syl_vits_1000ep':
        G_MODEL_PATH = "model/vits/syl_all_tsync_2/G_300000.pth"
        G_CONFIG_PATH = "configs/syl_thai_tsync.json"
    else:
        assert False, "exp_type is not match"
    # Add other conditions for different experiment types as needed

    return G_MODEL_PATH, G_CONFIG_PATH

# Entry point for the script
if __name__ == "__main__":
    G_MODEL_PATH, G_CONFIG_PATH = tts_path()
    main_process(G_MODEL_PATH, G_CONFIG_PATH)
