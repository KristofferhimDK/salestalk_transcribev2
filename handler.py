import runpod
import torch
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import tempfile
import os
import json

# Globale variabler - loader modellerne én gang når containeren starter
segmentation_pipeline = None
diarization_pipeline = None
whisper_pipe = None
device = None
torch_dtype = None

def init_models():
    """Initialiserer alle modeller - køres én gang ved opstart"""
    global segmentation_pipeline, diarization_pipeline, whisper_pipe, device, torch_dtype
    
    # HuggingFace authentication
    from huggingface_hub import login
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace login successful!")
    else:
        print("❌ Ingen HuggingFace token fundet!")
    
    print("🚀 Starter model initialisering...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"📱 Bruger device: {device}")
    
    # 1. Segmentering (dine modeller)
    print("📊 Loader segmentation model...")
    segmentation_pipeline = Pipeline.from_pretrained("syvai/speaker-segmentation")
    segmentation_pipeline = segmentation_pipeline.to(torch.device(device))
    
    # 2. Diarisering (opdateret med korrekte instruktioner)
    print("👥 Loader diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "syvai/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization_pipeline = diarization_pipeline.to(torch.device(device))
    
    # 3. Whisper til transskription (bruger deres instruktioner)
    print("🎤 Loader Whisper model...")
    model_id = "syvai/hviske-v3-conversation"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Opret ASR pipeline
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    
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
        
        # Transskriber dette segment med den nye pipeline
        result = whisper_pipe(segment_audio)
        transcription = result["text"]
        
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
