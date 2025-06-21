# --- START OF FILE app.py ---
#this file is a modified version of app.py which ignores everything but emotion detection part 
#it has emotion_detection hybrid vs tranformer vd rule based metric analysis with confusion matrix 
#no need to run this file, this file will be imported from emotion_detection.py for analysis just run that file only
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
from laion_clap import CLAP_Module # Import CLAP here as well
import time
import traceback
import tempfile # For safer temporary file handling

# --- Configuration (Global, outside UI function) ---
# Using session state for data that needs to persist across Streamlit reruns
if 'output_audio_path' not in st.session_state:
    st.session_state.output_audio_path = None
if 'generated_story_text' not in st.session_state:
    st.session_state.generated_story_text = None
if 'reference_voice_path' not in st.session_state:
    st.session_state.reference_voice_path = None

SOUND_DIR = "trimmed_sounds"
AUDIO_EMBEDDINGS_PATH = "audio_embeddings.pt"
TEMP_DIR = tempfile.gettempdir() # For uploaded reference voice
FINAL_OUTPUT_FILENAME_BASE = "final_story_output"

# --- Logging Setup (Global) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Cached Model Initializers & Helper Functions (Global) ---

@st.cache_data # Cache the fact that download was attempted/done
def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' tokenizer already available.")
    except LookupError:
        # Only show st.info if running within a Streamlit context
        try:
            if st.runtime.exists(): # Check if Streamlit is running
                st.info("Downloading NLTK 'punkt' tokenizer (one-time setup)...")
        except AttributeError: # Fallback for older Streamlit or non-Streamlit context
            pass
        logger.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK 'punkt' downloaded.")
        try:
            if st.runtime.exists():
                st.success("Tokenizer downloaded.")
        except AttributeError:
            pass

@st.cache_resource # Caches the actual model object
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
        try: # Try to show error in Streamlit if running
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
            try: # Try to show warning in Streamlit if running
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

# --- Core Logic Functions (Can be imported by other scripts) ---

def generate_story(prompt, status_placeholder=None):
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    fallback_story_text = """
        Pip, a small field mouse with a bright pink nose, tiptoed into the whispering woods. Crunch, crunch, crunch went his tiny paws on the fallen autumn leaves. Chirp, chirp, chirp sang the robins hidden amongst the branches. Pip stopped, his whiskers twitching. A distant Hoo-hoo echoed through the trees. A little scared, Pip peeked around a giant oak. The leaves rustled gently in the breeze like a soft whisper. Suddenly, Pip gasped! A beautiful clearing filled with glowing fireflies blinked before him. It was magical! He stood very still, filled with quiet wonder, his little heart thumping with happiness. The forest felt like a secret, just for him.
        """
    if not api_key_env:
        logger.warning("No Google API key found in environment. Using fallback story.")
        if status_placeholder: status_placeholder.warning("GOOGLE_API_KEY not found. Using fallback story.")
        return fallback_story_text
    try:
        genai.configure(api_key=api_key_env)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        if status_placeholder: status_placeholder.write("Generating story content via Gemini API...")
        logger.info("Generating story content...")
        response = model.generate_content(prompt)
        logger.info("Story content received from API.")
        if hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
            return response.parts[0].text
        elif hasattr(response, 'text'): return response.text
        else:
            feedback_info = f" Reason: {response.prompt_feedback}" if hasattr(response, 'prompt_feedback') else ""
            raise ValueError(f"Generation failed or blocked.{feedback_info}")
    except Exception as e:
        logger.error(f"Error generating story: {e}. Using fallback.")
        if status_placeholder: status_placeholder.error(f"Error generating story: {e}. Using fallback.")
        return fallback_story_text

def rule_based_emotion(sentence):
    text = sentence.lower()
    happy_words = ["happy", "joy", "excited", "laugh", "smile", "wonderful", "amazing", "delighted", "thrilled", "glee", "cheerful", "overjoyed", "enthusiastic", "best picnic ever", "magical", "glowing fireflies", "happiness"]
    if any(word in text for word in happy_words): return "happy"
    sad_words = ["sad", "cry", "tears", "unhappy", "miserable", "unfortunate", "sorry", "depressed", "gloomy", "heartbroken", "disappointed", "upset", "wept", "ruined", "sadly"]
    if any(word in text for word in sad_words): return "sad"
    angry_words = ["angry", "mad", "furious", "yell", "shout", "rage", "irritated", "frustrated", "enraged", "annoyed", "hate", "temper", "outraged", "how dare you"]
    if any(word in text for word in angry_words): return "angry"
    fear_words = ["scared", "afraid", "terrified", "fear", "frightened", "panic", "worried", "anxious", "dread", "horrified", "trembling", "hid under", "a little scared", "thunder", "storm", "lightning", "dark clouds", "hushed", "silenced"] # Added fear/tension words
    if any(word in text for word in fear_words): return "sad" # Map fear/tension to sad
    return None

def _run_transformer_classifiers(classifiers, sentence, confidence_threshold=0.4):
    if not classifiers:
        logger.warning("No ML emotion classifiers available for transformer processing, using neutral.")
        return "neutral"
    detected_emotions = []
    for name, classifier in classifiers.items():
        try:
            result_list = classifier(sentence)[0]
            if isinstance(result_list, list) and result_list:
                top_pred = max(result_list, key=lambda x: x['score'])
                emotion = top_pred["label"].lower()
                confidence = top_pred["score"]
                if confidence >= confidence_threshold: detected_emotions.append(emotion)
            else: logger.warning(f"Unexpected classifier output format for {name}: {result_list}")
        except Exception as e: logger.error(f"Error with classifier {name}: {e}")
    if not detected_emotions: return "neutral"
    emotion_mapping = {"joy": "happy", "happiness": "happy", "sadness": "sad", "anger": "angry", "fear": "sad", "surprise": "happy", "disgust": "angry", "neutral": "neutral", "love": "happy", "happy": "happy", "sad": "sad", "angry": "angry"}
    mapped_emotions = [emotion_mapping.get(e, "neutral") for e in detected_emotions]
    emotion_counts = Counter(mapped_emotions)
    return emotion_counts.most_common(1)[0][0]

def detect_emotion(classifiers, sentence, mode="hybrid"): # MODIFIED
    if mode == "rules_only":
        rule_emotion = rule_based_emotion(sentence)
        return rule_emotion if rule_emotion else "neutral"
    if mode == "transformer_only":
        return _run_transformer_classifiers(classifiers, sentence)
    if mode == "hybrid":
        rule_emotion = rule_based_emotion(sentence)
        if rule_emotion: return rule_emotion
        return _run_transformer_classifiers(classifiers, sentence)
    logger.warning(f"Unknown mode '{mode}' for detect_emotion. Defaulting to hybrid.")
    rule_emotion = rule_based_emotion(sentence) # Default to hybrid
    if rule_emotion: return rule_emotion
    return _run_transformer_classifiers(classifiers, sentence)

def apply_modulation(tts, text, emotion, reference_wav_path, output_path, status_placeholder=None):
    try:
        modulation = {"happy": {"energy": 1.05}, "sad": {"energy": 0.95}, "angry": {"energy": 1.0}, "neutral": {"energy": 1.0}}
        params = modulation.get(emotion, modulation["neutral"])
        if not tts: raise ValueError("TTS model not initialized")
        if not os.path.exists(reference_wav_path): raise FileNotFoundError(f"Reference WAV not found: {reference_wav_path}")
        
        output_dir_for_sentence = os.path.dirname(output_path)
        if output_dir_for_sentence and not os.path.exists(output_dir_for_sentence):
            os.makedirs(output_dir_for_sentence, exist_ok=True)

        tts.tts_to_file(text=text, speaker_wav=reference_wav_path, language="en", file_path=output_path)
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
        try: # Fallback
            if tts and os.path.exists(reference_wav_path):
                if status_placeholder: status_placeholder.warning("Attempting fallback neutral TTS...")
                tts.tts_to_file(text=text, speaker_wav=reference_wav_path, language="en", file_path=output_path)
                if os.path.exists(output_path): return output_path
            return None
        except Exception as fb_e:
            if status_placeholder: status_placeholder.error(f"Fallback TTS also failed: {fb_e}")
            return None

def process_audio_generation(
    tts, classifiers, clap_model, narration_text, reference_wav_path,
    audio_embeddings, sound_dir, # output_dir parameter removed, uses temp_run_dir internally
    final_output_path_base, similarity_thr, bg_db_reduction, pause_ms,
    status_placeholder, progress_bar ):

    if audio_embeddings and not clap_model:
        if status_placeholder: status_placeholder.warning("CLAP model not loaded. Background sounds disabled.")
    
    try:
        with tempfile.TemporaryDirectory(prefix="audio_gen_run_", dir=TEMP_DIR) as temp_run_dir:
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

            processed_segments = []
            emotion_summary = Counter()
            sound_match_summary = Counter()
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

                emotion = detect_emotion(classifiers, sentence, mode="hybrid") # Default hybrid for app
                emotion_summary[emotion] += 1
                current_sentence_status_parts.append(f"Emotion: {emotion}")
                logger.info(f"  Emotion: {emotion}")

                temp_tts_path = os.path.join(temp_run_dir, f"sentence_{i:03d}.wav")
                modulated_tts_path = apply_modulation(tts, sentence, emotion, reference_wav_path, temp_tts_path, status_placeholder)

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
                            text_embedding = clap_model.get_text_embedding([sentence], use_tensor=True)
                            if text_embedding is None: raise ValueError("Text embedding failed")
                            text_embedding = text_embedding.to(clap_device)
                            highest_similarity = -1
                            for sound_fname, audio_embedding in audio_embeddings.items():
                                audio_embedding_dev = audio_embedding.to(clap_device)
                                similarity = F.cosine_similarity(text_embedding, audio_embedding_dev, dim=1).item()
                                if similarity > highest_similarity:
                                    highest_similarity, best_match_sound = similarity, sound_fname
                        if best_match_sound and highest_similarity >= similarity_thr:
                            current_sentence_status_parts.append(f"Sound: '{os.path.basename(best_match_sound).split('.')[0]}' (Sim: {highest_similarity:.2f})")
                            sound_match_summary[best_match_sound] += 1
                        else: best_match_sound = None
                    except Exception as e_rag:
                        logger.error(f"  Error during sound matching for sentence {i+1}: {e_rag}", exc_info=False) # Less verbose exc_info for loop
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
                                if len(bg_segment) > 0: final_segment = tts_segment.overlay(bg_segment)
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
            # Output to current working directory if base is just filename
            final_output_dir = os.path.dirname(final_output_path_base) if os.path.dirname(final_output_path_base) else "." 
            final_output_basename = os.path.basename(final_output_path_base)
            os.makedirs(final_output_dir, exist_ok=True)
            final_output_path = os.path.join(final_output_dir, f"{final_output_basename}_{timestamp}.wav")

            if status_placeholder: status_placeholder.write(f"Exporting final audio to {final_output_path}...")
            combined_audio.export(final_output_path, format="wav")
            logger.info(f"Final audio saved: {final_output_path} - Duration: {len(combined_audio)/1000:.2f}s")
            if status_placeholder: status_placeholder.write("✅ **Stage: Merging Audio Segments Complete**")
            return final_output_path
    except Exception as e_main_proc:
        logger.error(f"Critical error in process_audio_generation: {e_main_proc}", exc_info=True)
        if status_placeholder: status_placeholder.error(f"Critical error during audio generation: {e_main_proc}")
        try:
            if st.runtime.exists(): st.exception(e_main_proc)
        except AttributeError: pass
        return None


# --- Function to Build Streamlit UI ---
def build_streamlit_ui():
    # Page setup (already done globally but can be here too if preferred for structure)
    # st.set_page_config(page_title="AI Audio Story Generator", layout="wide")
    # st.title("🎙️ AI Audio Story Generator with Background Sounds")
    st.markdown("""
    Create an audio narration for a story prompt using AI voice cloning and automatically added background sounds.
    Enter a prompt, upload a short voice sample, and let the AI craft your audio story!
    """)

    # Sidebar parameters (already defined globally using st.sidebar.slider)
    # Values are accessed directly via their global variable names:
    # similarity_threshold, background_db_reduction, pause_duration_ms

    # --- Inputs ---
    st.header("1. Inputs")
    col1, col2 = st.columns(2)
    with col1:
        prompt_text_ui = st.text_area("Enter Story Prompt:", height=150, placeholder="e.g., A brave knight faced a dragon...", key="prompt_text_key")
    with col2:
        uploaded_file_ui = st.file_uploader("Upload Reference Voice (.wav, ~5-30s):", type=['wav'], accept_multiple_files=False, key="uploaded_file_key")

    st.subheader("Optional: Personalize Your Story")
    child_name_ui = st.text_input("Child's Name (e.g., Lily):", key="child_name_ui_key")
    fav_animal_ui = st.text_input("Favorite Animal (e.g., bunny):", key="fav_animal_ui_key")
    fav_setting_ui = st.text_input("Favorite Setting (e.g., enchanted forest):", key="fav_setting_ui_key")

    # --- Generation Trigger ---
    st.header("2. Generate")
    generate_button = st.button("✨ Generate Story Audio ✨", type="primary", key="generate_button_key")

    # --- Progress & Output Area ---
    st.header("3. Results")
    progress_area = st.container()
    story_text_area_expander = st.expander("Generated Story Text", expanded=False)
    audio_player_area_main = st.container()

    # --- Main Logic on Button Press ---
    if generate_button:
        if not prompt_text_ui:
            st.warning("Please enter a story prompt.")
        elif uploaded_file_ui is None:
            st.warning("Please upload a reference WAV file.")
        else:
            st.session_state.output_audio_path = None
            st.session_state.generated_story_text = None
            st.session_state.reference_voice_path = None
            with story_text_area_expander: st.empty() # Clear previous text
            with audio_player_area_main: st.empty() # Clear previous player
            with progress_area: st.empty()

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR, prefix="ref_voice_") as tmp_ref_wav:
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
                    prompt_parts = [prompt_text_ui]
                    if child_name_ui: prompt_parts.append(f"The main character is named {child_name_ui}.")
                    if fav_animal_ui: prompt_parts.append(f"The story should feature a {fav_animal_ui}.")
                    if fav_setting_ui: prompt_parts.append(f"The story occurs in a {fav_setting_ui}.")
                    prompt_parts.append("Ensure a clear narrative arc and evoke some emotions. Make it suitable for audio.")
                    final_prompt_for_llm_ui = " ".join(prompt_parts)
                    
                    narration_text_generated = generate_story(final_prompt_for_llm_ui, overall_status)
                    if narration_text_generated:
                        st.session_state.generated_story_text = narration_text_generated
                        overall_status.write("✅ **Stage: Generating Story Text Complete**")
                        with story_text_area_expander: st.markdown(st.session_state.generated_story_text)
                        story_text_area_expander.expanded = True
                    else:
                        overall_status.error("Failed to generate story text.")
                        raise SystemExit("Story generation failed.")

                    progress_bar_ui = progress_area.progress(0.0, text="Initializing audio processing...")
                    final_audio_file_generated = process_audio_generation(
                        tts=tts_model_loaded, classifiers=emotion_classifiers_loaded,
                        clap_model=clap_model_loaded, narration_text=st.session_state.generated_story_text,
                        reference_wav_path=st.session_state.reference_voice_path,
                        audio_embeddings=audio_embeddings_loaded, sound_dir=SOUND_DIR,
                        final_output_path_base=FINAL_OUTPUT_FILENAME_BASE,
                        similarity_thr=similarity_threshold, bg_db_reduction=background_db_reduction,
                        pause_ms=pause_duration_ms, status_placeholder=overall_status,
                        progress_bar=progress_bar_ui
                    )

                    if final_audio_file_generated and os.path.exists(final_audio_file_generated):
                        overall_status.update(label="✅ Generation Complete!", state="complete", expanded=False)
                        st.session_state.output_audio_path = final_audio_file_generated
                    else:
                        overall_status.update(label="❌ Audio Generation Failed", state="error", expanded=True)
                        st.session_state.output_audio_path = None
                except SystemExit as se: # Catch explicit exits
                    logger.error(f"System exit during processing: {se}")
                    # Status already updated by the raising code usually
                except Exception as e_proc:
                    logger.error(f"Error in main processing block: {e_proc}", exc_info=True)
                    if 'overall_status' in locals() and overall_status:
                        overall_status.update(label="❌ Processing Failed", state="error", expanded=True)
                    st.exception(e_proc)
                finally:
                    if st.session_state.reference_voice_path and os.path.exists(st.session_state.reference_voice_path):
                        try:
                            os.remove(st.session_state.reference_voice_path)
                            logger.info(f"Cleaned up temporary reference file: {st.session_state.reference_voice_path}")
                            st.session_state.reference_voice_path = None
                        except Exception as cleanup_e:
                            logger.warning(f"Could not clean up temp file {st.session_state.reference_voice_path}: {cleanup_e}")

    # --- Display Final Results (runs on every rerun if path exists) ---
    if st.session_state.generated_story_text: # Show text if it was generated
        with story_text_area_expander:
            # This might re-write if the expander was cleared.
            # If already written by button logic, this is fine.
            st.markdown(st.session_state.generated_story_text)
        story_text_area_expander.expanded = True # Keep it expanded if text exists

    if st.session_state.output_audio_path:
        logger.info(f"Displaying audio player for: {st.session_state.output_audio_path}")
        with audio_player_area_main:
            st.subheader("🔊 Final Audio Story")
            st.markdown("*(Read the story text above while listening)*")
            try:
                if not os.path.exists(st.session_state.output_audio_path):
                    st.error("Generated audio file not found. Please try generating again.")
                    logger.error(f"Audio file not found for display: {st.session_state.output_audio_path}")
                else:
                    with open(st.session_state.output_audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                    st.download_button(
                        label="Download Audio Story (.wav)",
                        data=audio_bytes,
                        file_name=os.path.basename(st.session_state.output_audio_path),
                        mime='audio/wav'
                    )
            except Exception as e_display:
                st.error(f"Error displaying final audio: {e_display}")
                logger.error(f"Error reading/displaying audio file {st.session_state.output_audio_path}: {e_display}", exc_info=True)

# --- Guard for direct execution ---
if __name__ == "__main__":
    # This block only runs when app.py is executed directly as "streamlit run app.py"
    # It won't run when app.py is imported as a module.
    
    # Ensure NLTK data is ready (will use cache if already downloaded)
    download_nltk_punkt()
    
    build_streamlit_ui() # Call the function that creates and runs the Streamlit UI
# --- END OF FILE app.py ---
