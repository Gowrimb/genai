# --- START OF FILE merged_final.py ---

import streamlit as st
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment, silence
import nltk
from nltk.tokenize import sent_tokenize
from TTS.api import TTS
import google.generativeai as genai
from transformers import pipeline
import logging
from collections import Counter
import torch
import torch.nn.functional as F
from laion_clap import CLAP_Module
import time
import traceback
import tempfile
from dotenv import load_dotenv

load_dotenv()

# --- TEMPORARY DEBUG PRINT ---
loaded_api_key_debug = os.environ.get("GOOGLE_API_KEY")
if loaded_api_key_debug:
    print(f"DEBUG PRINT (merged_final.py): GOOGLE_API_KEY loaded: {loaded_api_key_debug[:5]}...{loaded_api_key_debug[-5:]}")
else:
    print("DEBUG PRINT (merged_final.py): GOOGLE_API_KEY NOT loaded from .env")
# --- END TEMPORARY DEBUG PRINT ---

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Audio Story Generator", layout="wide")
st.title("🎙️ AI Audio Story Generator with Background Sounds")
st.markdown("""
Create an audio narration for a story prompt using AI voice cloning and automatically added background sounds.
Enter a prompt, upload a short voice sample, and let the AI craft your audio story!
""")

# --- Configuration ---
if 'output_audio_path' not in st.session_state:
    st.session_state.output_audio_path = None
if 'generated_story_text' not in st.session_state:
    st.session_state.generated_story_text = None
if 'reference_voice_path' not in st.session_state:
    st.session_state.reference_voice_path = None

SOUND_DIR = "trimmed_sounds"
AUDIO_EMBEDDINGS_PATH = "audio_embeddings.pt"
TEMP_DIR_FOR_UPLOAD = tempfile.gettempdir()
FINAL_OUTPUT_FILENAME_BASE = "final_story_output"

# --- Sidebar for Parameters ---
st.sidebar.header("Configuration")
similarity_threshold = st.sidebar.slider(
    "Background Sound Similarity Threshold", 0.0, 1.0, 0.25, 0.01,
    help="How closely text must match sound embedding (higher means stricter match)."
)
background_db_reduction = st.sidebar.slider(
    "Background Sound Volume Reduction (dB)", -30, 0, -12, 1,
    help="How much quieter background sounds are relative to speech (more negative = quieter)."
)
pause_duration_ms = st.sidebar.slider(
    "Pause Between Sentences (ms)", 0, 1500, 400, 50,
    help="Duration of silence added between sentences."
)

# --- Model Loading (Cached) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' tokenizer already available.")
    except LookupError:
        try:
            if st.runtime.exists(): st.info("Downloading NLTK 'punkt' tokenizer (one-time setup)...")
        except AttributeError: pass
        logger.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK 'punkt' downloaded.")
        try:
            if st.runtime.exists(): st.success("Tokenizer downloaded.")
        except AttributeError: pass

@st.cache_resource
def load_tts_model():
    try:
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Initializing TTS model ({model_name})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = TTS(model_name).to(device)
        logger.info(f"XTTS model initialized and moved to target device: {device}")
        return tts_model
    except Exception as e:
        logger.error(f"Error initializing TTS ({model_name}): {e}", exc_info=True)
        try:
            if st.runtime.exists(): st.error(f"Fatal Error initializing TTS model: {e}")
        except AttributeError: pass
        return None

@st.cache_resource
def load_emotion_classifiers():
    classifiers = {}
    logger.info("Initializing emotion classifiers...")
    models = {
        "primary": "j-hartmann/emotion-english-distilroberta-base",
        "secondary": "bhadresh-savani/distilbert-base-uncased-emotion"
    }
    device_id = 0 if torch.cuda.is_available() else -1
    for name, model_name in models.items():
        try:
            logger.info(f"Loading classifier {name}: {model_name}")
            classifiers[name] = pipeline("text-classification", model=model_name, top_k=None, device=device_id)
            logger.info(f"Classifier {name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing emotion classifier {name} ({model_name}): {e}")
            try:
                if st.runtime.exists(): st.warning(f"Could not load emotion classifier {name} ({model_name}): {e}")
            except AttributeError: pass
    if not classifiers: logger.warning("No emotion classifiers could be initialized.")
    else: logger.info(f"Initialized {len(classifiers)} emotion classifier(s).")
    return classifiers if classifiers else None

@st.cache_resource
def load_clap_model_and_embeddings():
    clap_model_instance, audio_embeddings_data = None, None
    clap_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(AUDIO_EMBEDDINGS_PATH):
        logger.warning(f"Audio embeddings file not found: {AUDIO_EMBEDDINGS_PATH}. Background sounds disabled.")
        try:
            if st.runtime.exists(): st.warning(f"Audio embeddings file not found: {AUDIO_EMBEDDINGS_PATH}. Background sounds disabled.")
        except AttributeError: pass
        return None, None
    try:
        logger.info("Initializing CLAP model...")
        clap_model_instance = CLAP_Module(enable_fusion=False).to(clap_device)
        ckpt_path = os.path.expanduser("~/.cache/clap/630k-audioset-best.pt")
        if os.path.exists(ckpt_path):
            logger.info("Loading CLAP checkpoint...")
            try:
                clap_model_instance.load_ckpt(ckpt=ckpt_path)
                logger.info("CLAP checkpoint loaded successfully.")
                clap_model_instance.eval()
            except Exception as clap_load_e:
                logger.error(f"Failed to load CLAP checkpoint: {clap_load_e}", exc_info=True)
                try:
                    if st.runtime.exists(): st.error(f"Failed to load CLAP checkpoint: {clap_load_e}. Background sounds disabled.")
                except AttributeError: pass
                clap_model_instance = None
        else:
            logger.warning(f"CLAP checkpoint not found at {ckpt_path}. Background sounds disabled.")
            try:
                if st.runtime.exists(): st.warning(f"CLAP checkpoint not found: {ckpt_path}. Background sounds disabled.")
            except AttributeError: pass
            clap_model_instance = None
        if clap_model_instance:
            logger.info(f"Loading audio embeddings from {AUDIO_EMBEDDINGS_PATH}...")
            audio_embeddings_data = torch.load(AUDIO_EMBEDDINGS_PATH, map_location='cpu')
            logger.info(f"Loaded {len(audio_embeddings_data)} audio embeddings.")
        else: audio_embeddings_data = None
    except Exception as e:
        logger.error(f"Error during CLAP/Embedding initialization: {e}", exc_info=True)
        try:
            if st.runtime.exists(): st.error(f"Error during CLAP/Embedding initialization: {e}. Background sounds disabled.")
        except AttributeError: pass
        clap_model_instance, audio_embeddings_data = None, None
    return clap_model_instance, audio_embeddings_data

# --- Core Logic Functions ---

def generate_story(prompt, target_language_code, status_placeholder=None):
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    fallback_story_text = """
The old path crunched underfoot with many footsteps as Leo entered the deep forest. Suddenly, dark clouds gathered, and a gentle rain began to fall. A loud clap of thunder echoed, making the forest birds momentarily stop their chirping. Further on, Leo could hear the constant roar of distant ocean waves. The birds started singing again once the storm passed. He decided to head towards the sound of the sea.
"""
    if not api_key_env:
        logger.warning("No Google API key found in environment. Using fallback story.")
        if status_placeholder: status_placeholder.warning("GOOGLE_API_KEY not found. Using fallback story.")
        return fallback_story_text
    try:
        genai.configure(api_key=api_key_env)
        gemini_model_to_use = "gemini-1.5-flash-latest"
        logger.info(f"Attempting to use Gemini model: {gemini_model_to_use}")
        model = genai.GenerativeModel(gemini_model_to_use)

        full_prompt = f"{prompt} Please write this story in {target_language_code}."

        if status_placeholder: status_placeholder.write(f"Generating story content via Gemini ({gemini_model_to_use}) in {target_language_code}...")
        logger.info(f"Generating story content with {gemini_model_to_use} in {target_language_code}...")
        response = model.generate_content(full_prompt)
        logger.info("Story content received from API.")
        if hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
            return response.parts[0].text
        elif hasattr(response, 'text'): return response.text
        else:
            feedback_info = f" Reason: {response.prompt_feedback}" if hasattr(response, 'prompt_feedback') else ""
            raise ValueError(f"Generation failed or blocked.{feedback_info}")
    except Exception as e:
        current_model_name = gemini_model_to_use if 'gemini_model_to_use' in locals() else 'Gemini'
        logger.error(f"Error generating story with {current_model_name}: {e}. Using fallback.")
        if status_placeholder: status_placeholder.error(f"Error generating story with {current_model_name}: {e}. Using fallback.")
        return fallback_story_text

def detect_emotion(classifiers, sentence):
    """
    Detects emotion using loaded transformer-based classifiers only.
    Falls back to 'neutral' if no classifiers are available or if no strong emotion is detected.
    This function is taken from `final-transformer.py`.
    """
    if not classifiers:
        logger.warning("No emotion classifiers available. Defaulting to 'neutral' emotion.")
        return "neutral"

    detected_emotions = []
    confidence_threshold = 0.4 # Consider emotion only if confidence is above this threshold

    for name, classifier in classifiers.items():
        try:
            result_list = classifier(sentence)[0]
            if isinstance(result_list, list) and result_list:
                top_pred = max(result_list, key=lambda x: x['score'])
                emotion = top_pred["label"].lower()
                confidence = top_pred["score"]
                if confidence >= confidence_threshold:
                    detected_emotions.append(emotion)
            else:
                logger.warning(f"Unexpected classifier output format for {name}: {result_list}")
        except Exception as e:
            logger.error(f"Error with classifier {name} for sentence '{sentence[:50]}...': {e}")

    if not detected_emotions:
        return "neutral"

    # Map different labels to a consistent set of emotions
    emotion_mapping = {
        "joy": "happy", "happiness": "happy", "surprise": "happy", "love": "happy",
        "sadness": "sad", "fear": "sad",
        "anger": "angry", "disgust": "angry",
        "neutral": "neutral"
    }
    mapped_emotions = [emotion_mapping.get(e, "neutral") for e in detected_emotions]

    # Return the most common mapped emotion
    emotion_counts = Counter(mapped_emotions)
    return emotion_counts.most_common(1)[0][0]

def apply_modulation(tts, text, emotion, reference_wav_path, output_path, language_code, status_placeholder=None):
    try:
        modulation = {"happy": {"energy": 1.05}, "sad": {"energy": 0.95}, "angry": {"energy": 1.0}, "neutral": {"energy": 1.0}}
        params = modulation.get(emotion, modulation["neutral"])
        if not tts: raise ValueError("TTS model not initialized")
        if not os.path.exists(reference_wav_path): raise FileNotFoundError(f"Reference WAV not found: {reference_wav_path}")
        
        output_dir_for_sentence = os.path.dirname(output_path)
        if output_dir_for_sentence and not os.path.exists(output_dir_for_sentence):
            os.makedirs(output_dir_for_sentence, exist_ok=True)

        # Pass the language_code to tts_to_file
        tts.tts_to_file(text=text, speaker_wav=reference_wav_path, language=language_code, file_path=output_path)
        if os.path.exists(output_path):
            try:
                audio, sr = librosa.load(output_path, sr=None)
                if params["energy"] != 1.0: audio = audio * params["energy"]
                max_amp = np.max(np.abs(audio))
                if max_amp > 0.98: audio = audio * (0.98 / max_amp)
                sf.write(output_path, audio, sr)
                return output_path
            except Exception as post_proc_e:
                log_msg = f"Modulation post-processing failed for {os.path.basename(output_path)}: {post_proc_e}. Using original TTS."
                logger.error(log_msg)
                if status_placeholder: status_placeholder.warning(log_msg)
                return output_path
        else: raise RuntimeError(f"TTS failed to create file: {output_path}")
    except Exception as e:
        log_msg = f"Error applying modulation for '{text[:30]}...': {e}"
        logger.error(log_msg, exc_info=True)
        if status_placeholder: status_placeholder.error(log_msg)
        try: 
            if tts and os.path.exists(reference_wav_path):
                if status_placeholder: st.warning("Attempting fallback neutral TTS...")
                tts.tts_to_file(text=text, speaker_wav=reference_wav_path, language=language_code, file_path=output_path) # Fallback uses language_code
                if os.path.exists(output_path): return output_path
            return None
        except Exception as fb_e:
            if status_placeholder: st.error(f"Fallback TTS also failed: {fb_e}")
            return None

def process_audio_generation(
    tts, classifiers, clap_model, narration_text, reference_wav_path,
    audio_embeddings, sound_dir, 
    final_output_path_base, similarity_thr, bg_db_reduction, pause_ms,
    selected_language_code, # New parameter for language
    status_placeholder, progress_bar ):

    if audio_embeddings and not clap_model:
        if status_placeholder: status_placeholder.warning("CLAP model not loaded. Background sounds disabled.")
    
    try:
        with tempfile.TemporaryDirectory(prefix="audio_gen_run_", dir=TEMP_DIR_FOR_UPLOAD) as temp_run_dir: 
            if status_placeholder: status_placeholder.write(f"Using temporary directory for sentence files: {temp_run_dir}")
            logger.info(f"Using temporary run directory: {temp_run_dir}")

            narration_text = narration_text.replace('\x00', '').strip()
            sentences = sent_tokenize(narration_text)
            num_sentences = len(sentences)
            if num_sentences == 0:
                if status_placeholder: status_placeholder.error("Narration text resulted in zero sentences.")
                return None

            if status_placeholder: status_placeholder.write(f"⏳ **Stage: Processing Sentences** (Total: {num_sentences})")
            if progress_bar: progress_bar.progress(0.0, text="Initializing sentence processing...")

            processed_segments, emotion_summary, sound_match_summary = [], Counter(), Counter()
            clap_device = "cpu"
            if clap_model:
                try: clap_device = next(clap_model.parameters()).device
                except StopIteration: logger.warning("CLAP model has no parameters, using CPU for similarity.")
                except Exception as e_clap_dev: logger.warning(f"Could not get CLAP device: {e_clap_dev}, using CPU.")
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    if progress_bar: progress_bar.progress((i + 1) / num_sentences, text=f"Sentence {i+1}/{num_sentences} (Skipped empty)")
                    continue
                
                logger.info(f"Processing Sentence {i+1}/{num_sentences}: {sentence}")
                current_sentence_status_parts = [f"Sentence {i+1}: '{sentence[:30]}...'"]
                
                # Use the transformer-based detect_emotion
                emotion = detect_emotion(classifiers, sentence) 
                emotion_summary[emotion] += 1
                current_sentence_status_parts.append(f"Emotion: {emotion}")
                logger.info(f"  Emotion: {emotion}")

                temp_tts_path = os.path.join(temp_run_dir, f"sentence_{i:03d}.wav")
                # Pass the selected_language_code to apply_modulation
                modulated_tts_path = apply_modulation(tts, sentence, emotion, reference_wav_path, temp_tts_path, selected_language_code, status_placeholder)

                if not modulated_tts_path or not os.path.exists(modulated_tts_path):
                    logger.warning(f"  Skipping sentence {i+1} due to TTS/Modulation failure.")
                    if progress_bar: progress_bar.progress((i + 1) / num_sentences, text=f"Sentence {i+1}/{num_sentences} (TTS Failed)")
                    continue
                try:
                    tts_segment = AudioSegment.from_wav(modulated_tts_path)
                except Exception as e_load_seg:
                    logger.error(f"  Error loading TTS segment {os.path.basename(modulated_tts_path)}: {e_load_seg}. Skipping sentence.")
                    if progress_bar: progress_bar.progress((i + 1) / num_sentences, text=f"Sentence {i+1}/{num_sentences} (Load Failed)")
                    continue
                
                best_match_sound = None
                if clap_model and audio_embeddings:
                    try:
                        with torch.no_grad():
                            # --- ADDED/ENHANCED CLAP LOGGING ---
                            logger.debug(f"CLAP: Generating text embedding for: '{sentence[:50]}...'")
                            text_embedding = clap_model.get_text_embedding([sentence], use_tensor=True)
                            if text_embedding is None: 
                                logger.warning(f"CLAP: Text embedding failed for sentence: {sentence[:50]}...")
                                raise ValueError("Text embedding failed")
                            text_embedding = text_embedding.to(clap_device)
                            
                            highest_similarity = -1.0 # Initialize to a valid similarity value
                            temp_best_match_for_debug = "None"

                            for sound_fname, audio_embedding in audio_embeddings.items():
                                audio_embedding_dev = audio_embedding.to(clap_device)
                                similarity = F.cosine_similarity(text_embedding, audio_embedding_dev, dim=1).item()
                                # logger.debug(f"CLAP Compare: vs '{sound_fname}' -> Sim: {similarity:.4f}") # Optional: very verbose
                                if similarity > highest_similarity:
                                    highest_similarity = similarity
                                    best_match_sound = sound_fname # Actual variable used for decision
                                    temp_best_match_for_debug = sound_fname # For logging before threshold

                            log_msg_similarity = f"CLAP: Sentence '{sentence[:30]}...' - Highest sim: {highest_similarity:.3f} with '{os.path.basename(temp_best_match_for_debug if temp_best_match_for_debug else 'None')}' (Threshold: {similarity_thr:.2f})"
                            logger.info(log_msg_similarity)
                            if status_placeholder: status_placeholder.write(log_msg_similarity)
                            # --- END ADDED/ENHANCED CLAP LOGGING ---

                            if best_match_sound and highest_similarity >= similarity_thr:
                                # This part was for the progress bar text, already good
                                current_sentence_status_parts.append(f"Sound: '{os.path.basename(best_match_sound).split('.')[0]}' (Sim: {highest_similarity:.2f})")
                                sound_match_summary[best_match_sound] += 1
                                # Explicit log already added by the logger.info(log_msg_similarity) above
                            else: 
                                # Explicit log for no match above threshold
                                log_msg_no_match = f"CLAP: No sound match above threshold {similarity_thr:.2f} for sentence '{sentence[:30]}...'"
                                logger.info(log_msg_no_match)
                                if status_placeholder: status_placeholder.write(log_msg_no_match)
                                best_match_sound = None # Ensure it's None
                    except Exception as e_rag:
                        logger.error(f"  Error during sound matching for sentence {i+1}: {e_rag}", exc_info=False)
                        if status_placeholder: status_placeholder.warning(f"  Error during sound matching: {e_rag}")
                        best_match_sound = None
                
                final_segment = tts_segment
                if best_match_sound:
                    try:
                        bg_sound_path = os.path.join(sound_dir, best_match_sound)
                        if os.path.exists(bg_sound_path):
                            bg_segment = AudioSegment.from_wav(bg_sound_path) + bg_db_reduction
                            tts_dur, bg_dur = len(tts_segment), len(bg_segment)
                            if bg_dur > 0:
                                if bg_dur < tts_dur: bg_segment = (bg_segment * int(np.ceil(tts_dur / bg_dur)))[:tts_dur]
                                else: bg_segment = bg_segment[:tts_dur]
                                if len(bg_segment) > 0: 
                                    final_segment = tts_segment.overlay(bg_segment)
                                    # --- ADDED EXPLICIT LOG FOR MIXING ---
                                    log_msg_mixed = f"  Mixed with '{os.path.basename(best_match_sound)}' ({bg_db_reduction} dB)."
                                    logger.info(log_msg_mixed)
                                    if status_placeholder: status_placeholder.write(log_msg_mixed)
                                    # --- END ADDED EXPLICIT LOG ---
                        else: logger.warning(f"  Background sound file not found: {bg_sound_path}")
                    except Exception as e_mix: logger.error(f"  Error mixing sound {best_match_sound}: {e_mix}")
                
                processed_segments.append(final_segment)
                if pause_ms > 0: processed_segments.append(AudioSegment.silent(duration=pause_ms))
                
                if progress_bar: progress_bar.progress((i + 1) / num_sentences, text=" | ".join(current_sentence_status_parts))

            if status_placeholder: status_placeholder.write("✅ **Stage: Processing Sentences Complete**")
            if status_placeholder: status_placeholder.write("⏳ **Stage: Merging Audio Segments**")

            if not processed_segments:
                if status_placeholder: status_placeholder.error("No audio segments were processed.")
                return None
            if len(processed_segments) > 1 and processed_segments[-1].duration_seconds * 1000 == pause_ms and len(processed_segments[-1].get_array_of_samples()) == 0:
                processed_segments.pop()
            
            valid_segments = [seg for seg in processed_segments if isinstance(seg, AudioSegment)]
            if not valid_segments:
                if status_placeholder: status_placeholder.error("No valid audio segments to merge.")
                return None
            
            combined_audio = sum(valid_segments)
            timestamp = int(time.time())
            final_output_dir = os.path.dirname(final_output_path_base) if os.path.dirname(final_output_path_base) else "." 
            final_output_basename = os.path.basename(final_output_path_base)
            os.makedirs(final_output_dir, exist_ok=True)
            final_output_path = os.path.join(final_output_dir, f"{final_output_basename}_{timestamp}.wav")

            if status_placeholder: status_placeholder.write(f"Exporting final audio to {final_output_path}...")
            combined_audio.export(final_output_path, format="wav")
            logger.info(f"Final audio saved: {final_output_path} - Duration: {len(combined_audio)/1000:.2f}s")
            if status_placeholder: status_placeholder.write("✅ **Stage: Merging Audio Segments Complete**")
            
            # --- ADDED FINAL SUMMARY LOG TO CONSOLE ---
            logger.info("-" * 30)
            logger.info("Final Processing Summary:")
            logger.info(f"  Total Sentences: {num_sentences}")
            logger.info("  Emotion Counts:")
            for emotion, count in emotion_summary.items(): logger.info(f"    {emotion}: {count}")
            logger.info("  Background Sound Matches Used:")
            if sound_match_summary:
                for sound, count in sound_match_summary.items(): logger.info(f"    {os.path.basename(sound)}: {count}")
            else:
                logger.info("    No background sounds were matched and used.")
            logger.info("-" * 30)
            # --- END ADDED FINAL SUMMARY LOG ---
            return final_output_path
    except Exception as e_main_proc:
        logger.error(f"Critical error in process_audio_generation: {e_main_proc}", exc_info=True)
        if status_placeholder: st.error(f"Critical error during audio generation: {e_main_proc}")
        try: 
            if st.runtime.exists(): st.exception(e_main_proc)
        except AttributeError: pass
        return None


# --- Function to Build Streamlit UI ---
def build_streamlit_ui():
    st.markdown("""
    Create an audio narration for a story prompt using AI voice cloning and automatically added background sounds.
    Enter a prompt, upload a short voice sample, and let the AI craft your audio story!
    """)

    # XTTS V2 supported languages (as of last major update)
    # This list might need to be updated with the exact codes supported by your TTS library version
    # The language codes are usually ISO 639-1 (two-letter)
    xtts_supported_languages = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de",
        "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
        "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
        "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko",
        "Hindi": "hi", "Latvian": "lv", "Swedish": "sv" # Added a couple more common ones
    }
    
    st.header("1. Inputs")
    col1, col2 = st.columns(2)
    with col1:
        prompt_text_ui = st.text_area("Enter Story Prompt:", height=150, placeholder="e.g., A brave knight faced a dragon...", key="prompt_text_key")
        
        # New: Language selection
        selected_language_name = st.selectbox(
            "Select Story Language:",
            options=list(xtts_supported_languages.keys()),
            index=list(xtts_supported_languages.keys()).index("English"), # Default to English
            help="Choose the language for both the generated story text and the cloned voice."
        )
        selected_language_code = xtts_supported_languages[selected_language_name]


    with col2:
        uploaded_file_ui = st.file_uploader("Upload Reference Voice (.wav, ~5-30s):", type=['wav'], accept_multiple_files=False, key="uploaded_file_key")

    st.subheader("Optional: Personalize Your Story (in the selected language)")
    child_name_ui = st.text_input("Child's Name (e.g., Lily):", key="child_name_ui_key")
    fav_animal_ui = st.text_input("Favorite Animal (e.g., brave lion, fluffy bunny):", key="fav_animal_ui_key")
    fav_setting_ui = st.text_input("A Special Place (e.g., enchanted forest, sparkling beach):", key="fav_setting_ui_key")

    st.header("2. Generate")
    generate_button = st.button("✨ Generate Story Audio ✨", type="primary", key="generate_button_key")

    st.header("3. Results")
    progress_area = st.container()
    story_text_area_expander = st.expander("Generated Story Text", expanded=False)
    audio_player_area_main = st.container()

    if generate_button:
        if not prompt_text_ui:
            st.warning("Please enter a story prompt.")
        elif uploaded_file_ui is None:
            st.warning("Please upload a reference WAV file.")
        else:
            st.session_state.output_audio_path = None
            st.session_state.generated_story_text = None
            st.session_state.reference_voice_path = None 
            with story_text_area_expander: st.empty()
            with audio_player_area_main: st.empty()
            with progress_area: st.empty()

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR_FOR_UPLOAD, prefix="ref_voice_") as tmp_ref_wav:
                    tmp_ref_wav.write(uploaded_file_ui.getvalue())
                    st.session_state.reference_voice_path = tmp_ref_wav.name
                logger.info(f"Reference voice saved temporarily to: {st.session_state.reference_voice_path}")
            except Exception as e_save_ref:
                st.error(f"Failed to save uploaded reference voice: {e_save_ref}")
                st.stop()

            with progress_area.status("🚀 Starting generation process...", expanded=True) as overall_status:
                try:
                    overall_status.write("⏳ **Stage: Initializing AI Models** (TTS, Emotion, CLAP)...")
                    tts_model_loaded = load_tts_model()
                    emotion_classifiers_loaded = load_emotion_classifiers()
                    clap_model_loaded, audio_embeddings_loaded = load_clap_model_and_embeddings()
                    overall_status.write("✅ **Stage: Initializing AI Models Complete**")

                    if not tts_model_loaded:
                        overall_status.error("TTS Model failed to load. Cannot proceed.")
                        raise SystemExit("TTS Load Failure")

                    overall_status.write("⏳ **Stage: Constructing Prompt & Generating Story Text**...")
                    base_prompt_text = prompt_text_ui
                    prompt_parts = [base_prompt_text]
                    if child_name_ui: prompt_parts.append(f"The main character of this story is named {child_name_ui}.")
                    if fav_animal_ui: prompt_parts.append(f"The story should prominently feature a {fav_animal_ui}.")
                    if fav_setting_ui: prompt_parts.append(f"The story takes place in or around a {fav_setting_ui}.")
                    
                    # Instruct LLM to generate in the selected language
                    prompt_parts.append(f"Please write a short, engaging children's story based on these elements in {selected_language_name}.")
                    prompt_parts.append("Ensure the story has a clear narrative arc and evokes some emotions. Make it suitable for audio narration.")
                    final_prompt_for_llm_ui = " ".join(prompt_parts)
                    
                    logger.info(f"Final prompt for LLM: {final_prompt_for_llm_ui}")
                    overall_status.write(f"Using enriched prompt (first 100 chars): {final_prompt_for_llm_ui[:100]}...")
                    
                    # Pass the selected language code to generate_story
                    narration_text_generated = generate_story(final_prompt_for_llm_ui, selected_language_code, overall_status)
                    
                    if narration_text_generated and narration_text_generated.strip() and not narration_text_generated.startswith("(Fallback Story due to error)"):
                        st.session_state.generated_story_text = narration_text_generated
                        overall_status.write("✅ **Stage: Generating Story Text Complete**")
                        with story_text_area_expander: st.markdown(st.session_state.generated_story_text)
                        story_text_area_expander.expanded = True
                    else:
                        overall_status.error("Failed to generate story text or received fallback.")
                        if narration_text_generated and narration_text_generated.startswith("(Fallback Story due to error)"):
                             with story_text_area_expander: st.markdown(narration_text_generated)
                             story_text_area_expander.expanded = True
                        raise SystemExit("Story generation failed or used fallback.")

                    progress_bar_ui = progress_area.progress(0.0, text="Generating audio...")
                    
                    # Pass the selected language code to process_audio_generation
                    final_audio_output_path = process_audio_generation(
                        tts_model_loaded, emotion_classifiers_loaded, clap_model_loaded,
                        st.session_state.generated_story_text, st.session_state.reference_voice_path,
                        audio_embeddings_loaded, SOUND_DIR,
                        FINAL_OUTPUT_FILENAME_BASE, similarity_threshold, background_db_reduction, pause_duration_ms,
                        selected_language_code, # Pass the language code here
                        overall_status, progress_bar_ui
                    )

                    if final_audio_output_path:
                        st.session_state.output_audio_path = final_audio_output_path
                        overall_status.write("✅ **Story audio generation complete!**")
                        overall_status.update(state="complete")
                    else:
                        overall_status.error("Audio generation failed.")
                        overall_status.update(state="error")

                except SystemExit as se:
                    logger.error(f"Process stopped due to: {se}")
                    overall_status.update(state="error")
                    st.error("Generation stopped. Please check the logs/messages above for details.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
                    overall_status.error(f"An unexpected error occurred: {e}")
                    overall_status.update(state="error")
                    st.exception(e)
                finally:
                    # Cleanup temporary reference voice file
                    if st.session_state.reference_voice_path and os.path.exists(st.session_state.reference_voice_path):
                        try:
                            os.remove(st.session_state.reference_voice_path)
                            logger.info(f"Cleaned up temporary reference file: {st.session_state.reference_voice_path}")
                            st.session_state.reference_voice_path = None
                        except Exception as cleanup_e:
                            logger.warning(f"Could not clean up temp file {st.session_state.reference_voice_path}: {cleanup_e}")

    # Display results if available
    if st.session_state.output_audio_path and os.path.exists(st.session_state.output_audio_path):
        with audio_player_area_main:
            st.subheader("Generated Audio Story")
            st.audio(st.session_state.output_audio_path)
            st.download_button(
                label="Download Audio Story",
                data=open(st.session_state.output_audio_path, "rb").read(),
                file_name=os.path.basename(st.session_state.output_audio_path),
                mime="audio/wav"
            )
    elif generate_button: # Only show this if generate was clicked and failed
        with audio_player_area_main:
            st.info("Audio story will appear here once generated.")

# --- Run the UI ---
if __name__ == "__main__":
    download_nltk_punkt() # Ensure punkt tokenizer is downloaded
    build_streamlit_ui()

# --- END OF FILE merged_final.py ---