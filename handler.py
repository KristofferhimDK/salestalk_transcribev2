import runpod
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline
import tempfile
import os
import json

# Globale variabler - loader modellerne én gang når containeren starter
segmentation_pipeline = None
diarization_pipeline = None
whisper_processor = None
whisper_model = None
device = None

def init_models():
    """Initialiserer alle modeller - køres én gang ved opstart"""
    global segmentation_pipeline, diarization_pipeline, whisper_processor, whisper_model, device
    
    # HuggingFace authentication
    from huggingface_hub import login
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace login successful!")
    else:
        print("❌ Ingen HuggingFace token fundet!")
    
    print("🚀 Starter model initialisering...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Bruger device: {device}")
    
    # 1. Segmentering (dine modeller)
    print("📊 Loader segmentation model...")
    segmentation_pipeline = Pipeline.from_pretrained("syvai/speaker-segmentation")
    segmentation_pipeline = segmentation_pipeline.to(torch.device(device))
    
    # 2. Diarisering 
    print("👥 Loader diarization model...")
    diarization_pipeline = Pipeline.from_pretrained("syvai/speaker-diarization-3.1")
    diarization_pipeline = diarization_pipeline.to(torch.device(device))
    
    # 3. Whisper til transskription
    print("🎤 Loader Whisper model...")
    whisper_processor = WhisperProcessor.from_pretrained("syvai/hviske-v3-conversation")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("syvai/hviske-v3-conversation")
    whisper_model = whisper_model.to(device)
    
    print("✅ Alle modeller loaded!")

def segment_audio(audio_path):
    """Step 1: Segmenterer lyden i mindre stykker"""
    print("📊 Starter segmentering...")
    
    # Kør segmentation pipeline
    segmentation = segmentation_pipeline(audio_path)
    
    # Konverter til liste af segmenter
    segments = []
    for segment in segmentation:
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "duration": segment.end - segment.start
        })
    
    print(f"📊 Fandt {len(segments)} segmenter")
    return segments

def diarize_audio(audio_path):
    """Step 2: Finder hvem der taler hvornår"""
    print("👥 Starter diarisering...")
    
    # Kør diarization pipeline  
    diarization = diarization_pipeline(audio_path)
    
    # Konverter til liste af speakers
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "duration": turn.end - turn.start
        })
    
    print(f"👥 Fandt {len(set(s['speaker'] for s in speakers))} unikke speakers")
    return speakers

def transcribe_segments(audio_path, segments, speakers):
    """Step 3: Transskriberer hver persons tale"""
    print("🎤 Starter transskription...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    final_segments = []
    
    for segment in segments:
        # Find hvilken speaker der taler i dette segment
        segment_speaker = "UNKNOWN"
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        for speaker_info in speakers:
            # Check overlap mellem segment og speaker
            if (speaker_info["start"] <= segment_start <= speaker_info["end"] or
                speaker_info["start"] <= segment_end <= speaker_info["end"] or
                segment_start <= speaker_info["start"] <= segment_end):
                segment_speaker = speaker_info["speaker"]
                break
        
        # Extract audio for dette segment
        start_sample = int(segment_start * sr)
        end_sample = int(segment_end * sr)
        segment_audio = audio[start_sample:end_sample]
        
        # Transskriber dette segment
        inputs = whisper_processor(segment_audio, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(**inputs, language="da")
        
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        final_segments.append({
            "start": segment_start,
            "end": segment_end,
            "duration": segment["duration"],
            "speaker": segment_speaker,
            "text": transcription.strip()
        })
    
    print(f"🎤 Transskriberede {len(final_segments)} segmenter")
    return final_segments

def handler(job):
    """Hovedfunktionen som RunPod kalder"""
    try:
        # Få input fra RunPod job
        job_input = job["input"]
        audio_url = job_input.get("audio_url")
        
        if not audio_url:
            return {"error": "Ingen audio_url provided"}
        
        print(f"🎵 Behandler audio: {audio_url}")
        
        # Download audio til temp fil
        import requests
        response = requests.get(audio_url)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(response.content)
            temp_audio_path = temp_file.name
        
        try:
            # Step 1: Segmentering
            segments = segment_audio(temp_audio_path)
            
            # Step 2: Diarisering  
            speakers = diarize_audio(temp_audio_path)
            
            # Step 3: Transskription
            final_segments = transcribe_segments(temp_audio_path, segments, speakers)
            
            # Beregn statistikker
            total_duration = max(seg["end"] for seg in final_segments)
            speaker_stats = {}
            
            for segment in final_segments:
                speaker = segment["speaker"]
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "total_time": 0,
                        "word_count": 0,
                        "segments": 0
                    }
                
                speaker_stats[speaker]["total_time"] += segment["duration"]
                speaker_stats[speaker]["word_count"] += len(segment["text"].split())
                speaker_stats[speaker]["segments"] += 1
            
            # Tilføj percentages
            for speaker in speaker_stats:
                speaker_stats[speaker]["percentage"] = (
                    speaker_stats[speaker]["total_time"] / total_duration * 100
                )
            
            result = {
                "success": True,
                "total_duration": total_duration,
                "segments": final_segments,
                "speaker_stats": speaker_stats,
                "processing_info": {
                    "device": device,
                    "num_segments": len(final_segments),
                    "num_speakers": len(speaker_stats)
                }
            }
            
            print("✅ Processing færdig!")
            return result
            
        finally:
            # Clean up temp file
            os.unlink(temp_audio_path)
            
    except Exception as e:
        print(f"❌ Fejl: {str(e)}")
        return {"error": str(e)}

# Initialiser modeller når containeren starter
init_models()

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
