import streamlit as st
import torch
import torchaudio
import os
import base64
from audiocraft.models import MusicGen

@st.cache_resource
def load_model():
    try:
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_music_tensors(description, duration: int):
    st.info(f"Generating music for description: '{description}' and duration: {duration} seconds.")
    model = load_model()
    if model is None:
        return None

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    try:
        output = model.generate(
            descriptions=[description],
            progress=True,
            return_tokens=True
        )
        return output[0]
    except Exception as e:
        st.error(f"Error during music generation: {e}")
        return None

def save_audio(samples: torch.Tensor, sample_rate=32000, save_path="audio_output/"):
    if samples is None:
        st.error("No audio samples to save.")
        return

    os.makedirs(save_path, exist_ok=True)
    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon="🎵",
    page_title="Music Gen"
)

def main():
    st.title("Text to Music Generator 🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using the Music Gen Small model.")

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if text_area and time_slider:
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        if music_tensors is not None:
            save_audio(music_tensors)
            audio_filepath = 'audio_output/audio_0.wav'
            if os.path.exists(audio_filepath):
                with open(audio_filepath, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)
                    st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)
            else:
                st.error("Generated audio file not found.")
        else:
            st.error("Music generation failed.")

if __name__ == "__main__":
    main()
