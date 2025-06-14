import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import os

class SpeechRecognizer:
    def __init__(self, model_name="openai/whisper-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        return audio
    
    def transcribe(self, audio_path):
        """Transcribe audio file to text"""
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Process audio
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate token ids
        predicted_ids = self.model.generate(input_features)
        
        # Decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def transcribe_batch(self, audio_paths):
        """Transcribe multiple audio files"""
        transcriptions = []
        
        for audio_path in tqdm(audio_paths, desc="Transcribing"):
            try:
                transcription = self.transcribe(audio_path)
                transcriptions.append({
                    "file": audio_path,
                    "text": transcription
                })
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                transcriptions.append({
                    "file": audio_path,
                    "text": "Error in transcription"
                })
        
        return transcriptions
    
    def save_transcriptions(self, transcriptions, output_file):
        """Save transcriptions to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in transcriptions:
                f.write(f"File: {item['file']}\n")
                f.write(f"Transcription: {item['text']}\n\n")

def main():
    # Initialize speech recognizer
    recognizer = SpeechRecognizer()
    
    # Example usage
    print("Speech Recognition System initialized.")
    
    # Get list of audio files
    audio_dir = "audio_files"  # Replace with your audio directory
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not audio_files:
        print("No audio files found in the specified directory.")
        return
    
    # Transcribe files
    print(f"\nTranscribing {len(audio_files)} audio files...")
    transcriptions = recognizer.transcribe_batch(audio_files)
    
    # Save results
    output_file = "transcriptions.txt"
    recognizer.save_transcriptions(transcriptions, output_file)
    print(f"\nTranscriptions saved to {output_file}")

if __name__ == "__main__":
    main() 