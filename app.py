from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model configuration - maps dataset types to their model and preprocessor files
MODEL_CONFIG = {
    "kepler": {
        "model": "models/k2_logistic_model.joblib",
        "preprocessor": "preprocessors/k2_preprocessor.joblib"
    },
    "k2": {
        "model": "models/k2_logistic_model.joblib",
        "preprocessor": "preprocessors/k2_preprocessor.joblib"
    }
}

# Cache for loaded models
model_cache = {}


def load_model(dataset_type: str) -> Dict[str, Any]:
    """
    Load model and preprocessor for the specified dataset type.
    Uses caching to avoid reloading on every request.
    """
    if dataset_type not in MODEL_CONFIG:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: {list(MODEL_CONFIG.keys())}")
    
    # Check cache first
    if dataset_type in model_cache:
        logger.info(f"Using cached model for dataset type: {dataset_type}")
        return model_cache[dataset_type]
    
    # Load from disk
    config = MODEL_CONFIG[dataset_type]
    
    if not os.path.exists(config["model"]):
        raise FileNotFoundError(f"Model file not found: {config['model']}")
    if not os.path.exists(config["preprocessor"]):
        raise FileNotFoundError(f"Preprocessor file not found: {config['preprocessor']}")
    
    logger.info(f"Loading model for dataset type: {dataset_type}")
    model = joblib.load(config["model"])
    preprocessor = joblib.load(config["preprocessor"])
    
    # Cache the loaded model
    model_cache[dataset_type] = {
        "model": model,
        "preprocessor": preprocessor
    }
    
    return model_cache[dataset_type]


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        "status": "healthy",
        "service": "Exoplanet Classification API",
        "available_datasets": list(MODEL_CONFIG.keys())
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.
    
    Expected JSON format:
    {
        "dataset_type": "kepler" or "k2",
        "data": {
            "feature1": [value1, value2, ...],
            "feature2": [value1, value2, ...],
            ...
        }
        OR
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...}
        ]
    }
    
    Returns:
    {
        "predictions": ["CONFIRMED", "FALSE POSITIVE", ...],
        "probabilities": [[0.1, 0.9], [0.8, 0.2], ...],
        "dataset_type": "kepler"
    }
    """
    try:
        # Parse request
        try:
            request_data = request.get_json()
        except Exception as e:
            return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
        
        if not request_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_type = request_data.get("dataset_type", "").lower()
        
        data = request_data.get("data")
        
        if not dataset_type:
            return jsonify({"error": "Missing 'dataset_type' field"}), 400
        
        if data is None:
            return jsonify({"error": "Missing 'data' field"}), 400
        
        # Load the appropriate model and preprocessor
        try:
            model_components = load_model(dataset_type)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 500
        
        model = model_components["model"]
        preprocessor = model_components["preprocessor"]
        
        # Convert data to DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            return jsonify({"error": f"Failed to convert data to DataFrame: {str(e)}"}), 400
        
        if df.empty:
            return jsonify({"error": "Empty dataset provided"}), 400
        if dataset_type == "k2":
            cols_to_drop = ['pl_name', 'hostname', 'disp_refname', "disc_year",'pl_refname',
                'st_refname', 'sy_refname', 'rastr',"ra", 'decstr',"dec", 'rowupdate',
                'pl_pubdate', 'releasedate',
                "discoverymethod",  "soltype" #cheating
                , "disc_facility" # name of the facility meh,
                "disposition"
]       
        elif dataset_type == "kepler":
            cols_to_drop = ["kepid","kepoi_name", "kepler_name", "koi_time0bk", "koi_teq_err1", "koi_teq_err2",
                "koi_time0bk_err1", "koi_time0bk_err2", "koi_tce_plnt_num","koi_tce_delivname",
                "ra","dec" , 'koi_pdisposition' # time-series-analysis cheating
                , "koi_score" # score of cheat ,
                ,"koi_disposition"
                ]
        elif dataset_type =="tess":
            cols_to_drop = ["loc_rowid", "toi","tid", "rastr", "decstr",
                "toi_created", "rowupdate", "tfopwg_disp"
                ]
        cols_to_drop.extend([col for col in df.columns if "lim" in col])
        logger.info(f"Received prediction request for {len(df)} samples using {dataset_type} model")
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        
        print(df.columns)
        # Preprocess the data
        try:
            X_processed = preprocessor.transform(df)
        except Exception as e:
            return jsonify({
                "error": f"Preprocessing failed: {str(e)}",
                "hint": "Check that all required features are present"
            }), 400
        
        # Make predictions
        try:
            if dataset_type in  ["k2","kepler"]:
                # Probability of being "PLANET CANDIDATE"
                predictions = model.predict(X_processed)
                probabilities = model.predict_proba(X_processed)[:,0]
            else : 
                probabilities = model.predict(X_processed)
                predictions = (probabilities >= 0.5).astype(int)
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        # Format response
        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "dataset_type": dataset_type,
            "num_samples": len(predictions)
        }
        
        logger.info(f"Successfully predicted {len(predictions)} samples")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/models", methods=["GET"])
def list_models():
    """List available dataset types and their model status"""
    model_status = {}
    
    for dataset_type, config in MODEL_CONFIG.items():
        model_status[dataset_type] = {
            "model_exists": os.path.exists(config["model"]),
            "preprocessor_exists": os.path.exists(config["preprocessor"]),
            "cached": dataset_type in model_cache
        }
    
    return jsonify(model_status), 200


if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
