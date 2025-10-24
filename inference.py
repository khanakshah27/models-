import os
import joblib
import json
import spacy

# Load spaCy model once globally for preprocessing
nlp = spacy.load("en_core_web_sm")

def spacy_preprocess(texts):
    processed_texts = []
    for text in texts:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_punct]
        processed_texts.append(" ".join(tokens))
    return processed_texts

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data depending on content type."""
    if request_content_type == "application/json":
        input_json = json.loads(request_body)
        # Expect input_json to contain text data as a list or single string
        if isinstance(input_json, list):
            texts = input_json
        else:
            texts = [input_json]
        return texts
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run the inference."""
    # Preprocess text data using spaCy as done during training
    processed_texts = spacy_preprocess(input_data)
    predictions = model.predict(processed_texts)
    return predictions.tolist()

def output_fn(prediction, response_content_type):
    """Convert prediction output to desired content type."""
    if response_content_type == "application/json":
        return json.dumps({"predictions": prediction})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
