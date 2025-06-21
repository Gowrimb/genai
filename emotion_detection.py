# --- START OF FILE evaluate_emotion.py ---
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import nltk # For the NLTK download check
import logging # For consistency with app.py logging

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import necessary functions from app.py
# This assumes evaluate_emotion.py is in the same directory as app.py
try:
    from app1 import load_emotion_classifiers, detect_emotion, rule_based_emotion, _run_transformer_classifiers
    # Ensure NLTK is downloaded as app.py might do it within Streamlit context
    from app1 import download_nltk_punkt 
except ImportError as e:
    logger.error(f"Error importing from app1.py: {e}")
    logger.error("Ensure app1.py is in the current directory or Python path, and its Streamlit UI code is guarded by 'if __name__ == \"__main__\":'.")
    sys.exit(1)
except Exception as e_imp:
    logger.error(f"An unexpected error occurred during import from app1.py: {e_imp}")
    sys.exit(1)

EMOTION_LABELS = ["happy", "sad", "angry", "neutral"] # Consistent order for reports

def load_test_data(filepath="emotion_test_data.csv"):
    """Loads test data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        if "sentence" not in df.columns or "true_emotion" not in df.columns:
            raise ValueError("CSV must contain 'sentence' and 'true_emotion' columns.")
        # Normalize true labels (optional, but good for consistency)
        df["true_emotion"] = df["true_emotion"].str.lower().str.strip()
        logger.info(f"Loaded {len(df)} sentences from {filepath}")
        return df["sentence"].tolist(), df["true_emotion"].tolist()
    except FileNotFoundError:
        logger.error(f"Error: Test data file not found at {filepath}. Please create it.")
        return [], []
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return [], []

def run_evaluation():
    logger.info("--- Starting Emotion Detection Ablation Study ---")

    # Ensure NLTK data is downloaded (app.py's download_nltk_punkt uses @st.cache_data)
    # Calling it here will trigger the download if needed, and use cache if already done.
    # Note: Streamlit's st.info/st.success won't show in this console script.
    download_nltk_punkt()

    logger.info("Loading emotion classifiers...")
    # This will call the @st.cache_resource decorated function from app.py.
    # Streamlit's actual resource caching only works when run via `streamlit run`.
    # Here, it will just execute the function normally (load models from disk/HF).
    classifiers = load_emotion_classifiers() # This is the function from app.py
    
    if not classifiers:
        logger.error("Failed to load emotion classifiers. Cannot proceed with evaluation.")
        return

    logger.info("Loading test data (emotion_test_data.csv)...")
    sentences, true_labels = load_test_data()
    if not sentences:
        logger.error("No test data loaded. Please create 'emotion_test_data.csv'. Exiting.")
        return

    predictions = {
        "rules_only": [],
        "transformer_only": [],
        "hybrid": []
    }

    logger.info(f"Running predictions for {len(sentences)} sentences across different modes...")
    for i, sentence in enumerate(sentences):
        if (i + 1) % 20 == 0 or i == len(sentences) -1 : # Log progress every 20 sentences and for the last one
            logger.info(f"  Processing sentence {i+1}/{len(sentences)}")
        
        # Call the modified detect_emotion from app.py
        predictions["rules_only"].append(detect_emotion(classifiers, sentence, mode="rules_only"))
        predictions["transformer_only"].append(detect_emotion(classifiers, sentence, mode="transformer_only"))
        predictions["hybrid"].append(detect_emotion(classifiers, sentence, mode="hybrid"))

    # Generate and print reports
    logger.info("\n--- Evaluation Results ---")
    for mode, pred_labels in predictions.items():
        print(f"\n\n===== Results for: {mode.upper()} =====") # Using print for clearer console output
        
        if len(true_labels) != len(pred_labels):
            logger.warning(f"Length mismatch for {mode}: True labels={len(true_labels)}, Pred labels={len(pred_labels)}. Skipping report.")
            continue

        report_str = classification_report(true_labels, pred_labels, labels=EMOTION_LABELS, zero_division=0, output_dict=False)
        print(report_str)

        cm = confusion_matrix(true_labels, pred_labels, labels=EMOTION_LABELS)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot confusion matrix
        try:
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, cmap="Blues")
            plt.title(f"Confusion Matrix - {mode.upper()}", fontsize=14)
            plt.ylabel("Actual Emotion", fontsize=12)
            plt.xlabel("Predicted Emotion", fontsize=12)
            plt.tight_layout()
            plot_filename = f"confusion_matrix_{mode}.png"
            plt.savefig(plot_filename)
            plt.close() # Close the plot to free memory
            logger.info(f"Saved confusion matrix plot to ./{plot_filename}")
        except Exception as plot_e:
            logger.error(f"Could not plot/save confusion matrix for {mode}: {plot_e}")

    logger.info("--- Evaluation Complete ---")

if __name__ == "__main__":
    run_evaluation()
# --- END OF FILE evaluate_emotion.py ---