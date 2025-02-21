import queue
import pyaudio
import numpy as np
import torch
import os
import keyboard
import warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline

# Suppress all warnings from transformers
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Stable Diffusion Model
print("🎨 Loading Stable Diffusion model...")
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
print("✅ Stable Diffusion loaded successfully!")

class SpeechToText:
    def __init__(self):  
        """Initialize Speech to Text"""
        # Configuration
        self.SAMPLE_RATE = 16000
        self.FRAMES_PER_BUFFER = 3200
        
        # Initialize audio and model
        print("🎤 Setting up audio recording...")
        self.setup_audio()
        print("✅ Audio setup complete!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.setup_model()
        
        # State variables
        self.is_recording = False

    def setup_audio(self):
        """Initialize audio recording components"""
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(
            frames_per_buffer=self.FRAMES_PER_BUFFER,
            rate=self.SAMPLE_RATE,
            format=pyaudio.paInt16,
            channels=1,
            input=True
        )

    def setup_model(self):
        """Initialize Whisper model"""
        print("🔍 Loading Whisper model...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')

            if torch.cuda.is_available():
                print("🚀 Using GPU for faster processing...")
                self.whisper_model = self.whisper_model.to("cuda")

        print("✅ Whisper model loaded successfully!")

    def record_audio(self):
        """Record audio from the microphone until 'q' is pressed"""
        self.is_recording = True
        audio_frames = []

        print("\n🎤 Recording in progress... Press 'q' to stop.")

        while self.is_recording:
            try:
                data = self.audio_stream.read(self.FRAMES_PER_BUFFER, exception_on_overflow=False)
                audio_frames.append(data)

                if keyboard.is_pressed('q'):
                    print("\n🛑 Recording stopped.")
                    self.is_recording = False
                    break

            except Exception as e:
                print(f"❌ Audio recording error: {e}")
                break

        return audio_frames

    def process_audio(self, audio_data):
        """Process audio through Whisper model and force English transcription"""
        try:
            # Convert audio data to numpy array
            audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Process audio with English language enforcement
                inputs = self.whisper_processor(
                    audio_np,
                    sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt",
                    language="en"  # ✅ Force Whisper to transcribe only in English
                )

                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")

                # Generate transcription
                predicted_ids = self.whisper_model.generate(inputs.input_features)
                
                # Decode the output
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

            return transcription.strip()
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return ""

    def run(self):
        """Main execution loop"""
        print("\n=== Speech to Text Converter ===")
        print("Press 's' to start recording, 'q' to stop recording, and Ctrl+C to exit")

        try:
            while True:
                if keyboard.is_pressed('s'):
                    print("\n🎤 Starting recording...")
                    audio_data = self.record_audio()
                    
                    if audio_data:
                        print("\n🔄 Processing audio...")
                        transcription = self.process_audio(audio_data)
                        
                        if transcription:
                            print(f"\n📝 Transcription: {transcription}\n")

                            # Generate image from text
                            prompt = transcription
                            image = pipe(prompt).images[0]
                            image.save("./generated_image.png")

                            print("✅ Image saved as 'generated_image.png'.")

                        else:
                            print("\n⚠ No speech detected or processing failed.")
                            
        except KeyboardInterrupt:
            print("\n🚪 Exiting program...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.p.terminate()
        print("🧹 Resources cleaned up.")

if __name__ == "__main__":
    converter = SpeechToText()
    converter.run()
