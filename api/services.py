from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
import pandas as pd
from numpy import ndarray
import joblib
from typing import List, Dict

class Services:
    """
    A service class for encoding text and categorizing based on embeddings.
    
    Attributes:
        model (SentenceTransformer): Preloaded SBERT model.
        classifier (joblib): Preloaded KNN classifier for topic categorization.
    """
    
    model = None
    classifier = None
    label_encoder = None

    @classmethod
    def initialize_models(cls, model_name: str = "paraphrase-MiniLM-L6-v2",
                          version: str = "latest"):
        """
        Initializes and caches the SBERT model, classifier, and label encoder.
        
        Args:
            model_name (str): Name of the SBERT model.
            version (str): Version of the KNN model and label encoder to load.
        """
        # Define versioned paths for the classifier and encoder
        classifier_path = f"../models/{version}/model_knn.pkl"
        encoder_path = f"../models/{version}/label_encoder.pkl"
        
        if cls.model is None:
            cls.model = SentenceTransformer(model_name)
        if cls.classifier is None:
            cls.classifier = joblib.load(classifier_path)
        if cls.label_encoder is None:
            cls.label_encoder = joblib.load(encoder_path)

    @staticmethod
    async def encode(texts: List[str]) -> List[List[float]]:
        """
        Computes SBERT embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to encode.
        
        Returns:
            List[List[float]]: List of embedding vectors for each text.
        
        Raises:
            HTTPException: If encoding fails.
        """
        try:
            # Ensure models are initialized
            Services.initialize_models(version="v1")
            vectors: ndarray = Services.model.encode(sentences=texts, show_progress_bar=False)
            return vectors.tolist()
        except Exception as e:
            raise HTTPException(500, f"Encoding failed due to an internal error: {e}")

        
    @staticmethod
    async def get_topics_by_threshold(vector: List[List[float]], threshold: float = 0.2) -> List[Dict[str, float]]:
        """
        Predicts topic categories with confidence scores above a specified threshold.

        Args:
            vector (List[List[float]]): List of embedding vectors for categorization.
            threshold (float): Probability threshold for including topics in the results.
        
        Returns:
            List[Dict[str, float]]: List of dictionaries with topic labels and their probabilities.
        """
        try:
            Services.initialize_models()
            vector_df = pd.DataFrame(vector)
            vector_df.columns = [f"dim_{i}" for i in range(vector_df.shape[1])]

            # Get probabilities for each class
            probabilities = Services.classifier.predict_proba(vector_df)[0]
            
            # Filter topics based on the threshold
            topics_above_threshold = [
                {
                    "topic": Services.label_encoder.inverse_transform([index])[0],
                    "confidence": round(prob, 4)
                }
                for index, prob in enumerate(probabilities) if prob >= threshold
            ]

            # Sort topics by confidence in descending order
            topics_above_threshold.sort(key=lambda x: x["confidence"], reverse=True)
            return topics_above_threshold
        except Exception as e:
            raise HTTPException(500, f"Categorization failed due to an internal error: {e}")