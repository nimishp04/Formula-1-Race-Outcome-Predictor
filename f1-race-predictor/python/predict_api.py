"""
F1 Race Predictor - Flask API with Trained Neural Network
Real-time predictions using the trained ML model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Global variables for loaded models
model = None
scaler = None
imputer = None
feature_names = None
model_info = None

def load_models():
    """Load trained models and preprocessing objects"""
    global model, scaler, imputer, feature_names, model_info
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    print("Loading trained models...")
    
    # Load Neural Network (best model)
    with open(models_dir / 'neural_network_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Neural Network loaded")
    
    # Load scaler
    with open(models_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
    
    # Load imputer (if exists)
    imputer_path = models_dir / 'imputer.pkl'
    if imputer_path.exists():
        with open(imputer_path, 'rb') as f:
            imputer = pickle.load(f)
        print("✓ Imputer loaded")
    
    # Load feature names
    with open(models_dir / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✓ Feature names loaded ({len(feature_names)} features)")
    
    # Load model info
    with open(models_dir / 'model_info.json', 'r') as f:
        model_info = json.load(f)
    print("✓ Model info loaded")
    
    print(f"\n🎯 Model: Neural Network")
    print(f"   Accuracy: {model_info['models']['Neural Network']['accuracy']:.4f}")
    print(f"   Trained on: {model_info['dataset']['train_samples']} samples")
    print("\nAPI Ready! 🚀\n")

def create_feature_vector(input_data):
    """
    Create 27-feature vector from user input
    Matches the training feature engineering
    """
    
    # Map simple inputs to full feature set
    # Since we don't have full history in real-time, use approximations
    
    grid = input_data['gridPosition']
    driver_avg = input_data['driverAvgPos']
    recent_wins = input_data['recentWins']
    recent_podiums = input_data['recentPodiums']
    constructor_avg = input_data['constructorAvgPos']
    circuit_avg = input_data['circuitAvgPos']
    fastest_lap_rate = input_data.get('fastestLapRate', 30) / 100
    finish_rate = input_data.get('finishRate', 95) / 100
    
    # Create full 27-feature vector
    # These match the features in training
    features = {
        # Grid & Qualifying
        'grid_position': grid,
        'grid_position_squared': grid ** 2,
        
        # Driver Recent Form (Last 5 races) - approximated
        'driver_avg_position_5': driver_avg,
        'driver_best_position_5': max(1, driver_avg - 1),
        'driver_worst_position_5': min(20, driver_avg + 2),
        'driver_position_std_5': 2.0,  # Reasonable default
        
        # Driver Medium Term (Last 10 races) - approximated
        'driver_avg_position_10': driver_avg,
        'driver_wins_10': recent_wins * 2,  # Scale up for 10 races
        'driver_podiums_10': recent_podiums * 2,
        'driver_top5_10': min(10, recent_podiums * 2 + 2),
        'driver_points_avg_10': max(0, (20 - driver_avg) * 2),
        
        # Win/Podium rates
        'driver_win_rate': recent_wins / 5,
        'driver_podium_rate': recent_podiums / 5,
        
        # DNF rate (reliability)
        'driver_dnf_rate': 1 - finish_rate,
        
        # Constructor Form
        'constructor_avg_position': constructor_avg,
        'constructor_wins': int(recent_wins * 1.5),  # Team typically has 2 drivers
        'constructor_podiums': int(recent_podiums * 1.5),
        'constructor_points_avg': max(0, (20 - constructor_avg) * 3),
        
        # Circuit-Specific Performance
        'circuit_avg_position': circuit_avg,
        'circuit_best_position': max(1, circuit_avg - 1),
        'circuit_races': 3,  # Assume some circuit experience
        'circuit_wins': 1 if circuit_avg <= 2 else 0,
        
        # Momentum indicators
        'recent_trend': 0,  # Neutral default
        'improving': 1 if driver_avg <= 5 else 0,
        
        # Consistency
        'consistency_score': 1 / (2 + 1),  # Moderate consistency
        
        # Season context
        'round_number': input_data.get('roundNumber', 10),
        'season_progress': input_data.get('roundNumber', 10) / 20,
    }
    
    # Convert to DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Reorder to match training feature order
    df = df[feature_names]
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict win probability using trained Neural Network
    
    Expected JSON input:
    {
        "gridPosition": 1,
        "driverAvgPos": 2.5,
        "recentWins": 3,
        "recentPodiums": 4,
        "constructorAvgPos": 1.5,
        "circuitAvgPos": 2.0,
        "fastestLapRate": 40,
        "finishRate": 95,
        "roundNumber": 10
    }
    """
    try:
        # Get input data
        input_data = request.json
        
        # Validate required fields
        required_fields = ['gridPosition', 'driverAvgPos', 'recentWins', 
                          'recentPodiums', 'constructorAvgPos', 'circuitAvgPos']
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create feature vector
        features_df = create_feature_vector(input_data)
        
        # Handle NaN values if imputer exists
        if imputer is not None:
            features = imputer.transform(features_df)
        else:
            features = features_df.values
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        win_probability = model.predict_proba(features_scaled)[0][1]
        prediction = int(model.predict(features_scaled)[0])
        
        # Calculate podium probabilities (simplified)
        grid = input_data['gridPosition']
        podiums = input_data['recentPodiums']
        
        # Estimate P2 and P3 probabilities
        if grid <= 6:
            p2_prob = max(0, 0.35 - grid * 0.04 + (podiums * 0.03))
            p3_prob = max(0, 0.30 - grid * 0.03 + (podiums * 0.02))
        elif grid <= 10:
            p2_prob = max(0, 0.20 - grid * 0.02)
            p3_prob = max(0, 0.25 - grid * 0.02)
        else:
            p2_prob = max(0, 0.10 - grid * 0.01)
            p3_prob = max(0, 0.15 - grid * 0.01)
        
        total_podium = min(win_probability + p2_prob + p3_prob, 0.95)
        
        # Return prediction
        return jsonify({
            'success': True,
            'model': 'Neural Network',
            'accuracy': model_info['models']['Neural Network']['accuracy'],
            'predictions': {
                'win_probability': float(win_probability),
                'win_probability_percent': round(float(win_probability) * 100, 2),
                'will_win': bool(prediction),
                'podium': {
                    'p1': round(float(win_probability) * 100, 2),
                    'p2': round(p2_prob * 100, 2),
                    'p3': round(p3_prob * 100, 2),
                    'any_podium': round(total_podium * 100, 2)
                }
            },
            'input_received': input_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model': 'Neural Network',
        'info': model_info['models']['Neural Network'],
        'dataset': model_info['dataset'],
        'features': len(feature_names),
        'feature_list': feature_names
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'F1 Race Predictor API',
        'version': '2.0',
        'model': 'Neural Network (95.54% accuracy)',
        'endpoints': {
            'POST /predict': 'Make race outcome prediction',
            'GET /model-info': 'Get model information',
            'GET /health': 'Health check'
        },
        'example_request': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'gridPosition': 1,
                'driverAvgPos': 2.5,
                'recentWins': 3,
                'recentPodiums': 4,
                'constructorAvgPos': 1.5,
                'circuitAvgPos': 2.0,
                'fastestLapRate': 40,
                'finishRate': 95
            }
        }
    })

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run Flask app
    print("="*70)
    print("Starting F1 Race Predictor API Server...")
    print("="*70)
    print("\nAPI running at: http://localhost:5000")
    print("Try it: http://localhost:5000/health\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)