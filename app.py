import whisper
import streamlit as st
import av
import torch
import tempfile
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import os

st.set_page_config(page_title="🎙️ Voice to Text Transcriber", layout="centered")
st.title("🎙️ Voice to Text (Mic Input) with Whisper")
st.markdown("Record your voice using the mic and get the transcribed text using OpenAI's Whisper model.")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# Audio processor for mic recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Collect audio frames
        self.recorded_frames.append(frame)
        return frame

st.markdown("### Step 1: Record your voice")

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode="sendrecv",
    in_audio=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=AudioProcessor,
)

# Save audio when done
if webrtc_ctx and webrtc_ctx.audio_processor:
    if st.button("🔴 Stop and Transcribe"):
        processor = webrtc_ctx.audio_processor
        frames = processor.recorded_frames
        if not frames:
            st.warning("No audio recorded.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                # Convert to AudioSegment
                audio = AudioSegment.empty()
                for frame in frames:
                    samples = frame.to_ndarray().flatten().astype("int16").tobytes()
                    segment = AudioSegment(
                        samples,
                        frame_rate=frame.sample_rate,
                        sample_width=2,
                        channels=1
                    )
                    audio += segment
                audio.export(f.name, format="wav")
                file_path = f.name

            st.info("Transcribing...")

            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)

            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)

            st.success("Transcription complete!")
            st.markdown(f"**Detected Language:** `{lang}`")
            st.text_area("📝 Transcribed Text", result.text, height=150)

            os.remove(file_path)
