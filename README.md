# Formula 1 Race Outcome Predictor

Neural network system for predicting F1 race winners with 95.54% accuracy using 75 years of historical data.

## Overview

Machine learning pipeline that predicts Formula 1 race outcomes using a Multi-Layer Perceptron neural network trained on 26,759 race results from 1950-2024.

**Performance**: 95.54% accuracy | ROC-AUC 0.9457 | 27 engineered features

## Project Workflow

### Phase 1: Data Processing → JSON Databases

```bash
python python/process_data.py
```

**Input**: Raw CSV files from Kaggle (races.csv, results.csv, drivers.csv, etc.)

**Process**:
- `F1DataProcessor` class loads 6 CSV files
- Calculates aggregate statistics (win rates, career totals, podium percentages)
- Analyzes grid position win rates (P1=42.34%, P2=23.82%, etc.)

**Output**: 4 JSON databases
- `data/drivers.json` - 862 drivers with career stats
- `data/constructors.json` - 212 teams with performance metrics
- `data/circuits.json` - 77 circuits with race history
- `data/historical.json` - Aggregated statistics and top performers

---

### Phase 2: Feature Engineering & Model Training → Trained Models

```bash
python python/train_ml_models.py
```

**Input**: Raw CSV files (races, results, qualifying, standings)

**Process**:

**Step 1 - Feature Engineering** (`load_and_engineer_features()`):
- Merges datasets chronologically by race date
- For each of 26,759 race results:
  - Looks back in time only (`merged.index < idx` prevents leakage)
  - Gets driver's last 10 races, constructor's last 10 races, circuit history
  - Computes 27 features: grid position, recent form, win rates, team performance, momentum

**Step 2 - Train/Test Split** (`prepare_train_test_split()`):
- Temporal split: Train on 1950-2018 (25,107 samples), Test on 2019-2024 (2,558 samples)
- Preprocessing: SimpleImputer (median) → StandardScaler (z-normalization)

**Step 3 - Train 5 Models**:
- XGBoost, Random Forest, Gradient Boosting, Neural Network, Logistic Regression
- Each model trained on scaled features, evaluated on test set

**Step 4 - Compare and Select** (`compare_models()`):
- Evaluates accuracy, precision, recall, F1-score, ROC-AUC
- Neural Network wins: 95.54% accuracy, 0.9457 ROC-AUC

**Output**: Trained model artifacts in `models/` folder
- `neural_network_model.pkl` - MLP weights (128-64-32 architecture)
- `scaler.pkl` - StandardScaler for preprocessing
- `imputer.pkl` - SimpleImputer for missing values
- `feature_names.txt` - List of 27 features in order
- `model_info.json` - Accuracy, training date, dataset info
- `model_comparison.csv` - Performance of all 5 models

---

### Phase 3: Production API → Real-Time Predictions

```bash
python python/predict_api.py
# API runs at http://localhost:5000
```

**Startup** (`load_models()`):
- Loads `neural_network_model.pkl`, `scaler.pkl`, `imputer.pkl` from disk
- Loads `feature_names.txt` (27 features), `model_info.json` (metadata)
- Prints model accuracy and "API Ready!"

**Inference** (`POST /predict`):
1. Receives JSON with simplified inputs (gridPosition, recentWins, driverAvgPos, etc.)
2. `create_feature_vector()` approximates full 27 features:
   - Scales 5-race stats to 10-race: `driver_wins_10 = recentWins × 2`
   - Uses defaults where data unavailable: `driver_position_std_5 = 2.0`
3. Applies imputer → scaler (same preprocessing as training)
4. Neural network forward pass: `model.predict_proba(features_scaled)`
5. Returns JSON with win probability, podium predictions

**Other Endpoints**:
- `GET /model-info` - Returns accuracy, feature list, dataset info
- `GET /health` - Checks if model loaded successfully
- `GET /` - API documentation

---

### Phase 4: Web Interface → User Interaction

Open `index_ml.html` in browser (no installation needed)

**Features**:
- Multi-driver comparison mode
- Race environment settings (track temp, rain probability, circuit type, weather)
- Real-time predictions via JavaScript fetch() to API
- Displays podium predictions and racing insights
- Fallback mode if API unavailable (uses heuristic formula)

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/[your-username]/f1-race-predictor.git
cd f1-race-predictor

# 2. Install dependencies
pip install -r python/requirements.txt

# 3. Download dataset from Kaggle and place CSVs in data_raw/

# 4. Process data
python python/process_data.py

# 5. (Optional) Train models - or use pre-trained models
python python/train_ml_models.py

# 6. Start API
python python/predict_api.py

# 7. Open index_ml.html in browser
```

## Model Architecture

**Multi-Layer Perceptron** (scikit-learn MLPClassifier):
```
27 features → 128 neurons (ReLU) → 64 neurons (ReLU) → 32 neurons (ReLU) → 2 outputs (Softmax)
```

**Hyperparameters**: activation=relu, solver=adam, alpha=0.0001, batch_size=32, early_stopping=True

## 27 Features

| Category | Count | Examples |
|----------|-------|----------|
| Grid & Qualifying | 2 | grid_position, grid_position² |
| Driver Recent (5 races) | 4 | avg_position_5, wins, podiums, std |
| Driver Medium (10 races) | 5 | avg_position_10, wins_10, podiums_10, top5_10, points |
| Win/Podium Rates | 3 | win_rate, podium_rate, dnf_rate |
| Constructor | 4 | constructor_avg, wins, podiums, points |
| Circuit-Specific | 4 | circuit_avg, circuit_best, circuit_wins, circuit_races |
| Momentum | 2 | recent_trend, improving |
| Consistency | 1 | consistency_score |
| Season Context | 2 | round_number, season_progress |

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Neural Network** | **95.54%** | **0.9457** |
| XGBoost | 92.83% | 0.9036 |
| Random Forest | 91.92% | 0.9420 |
| Gradient Boosting | 89.02% | 0.9300 |
| Logistic Regression | 84.00% | 0.9451 |

## API Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gridPosition": 1,
    "driverAvgPos": 2.5,
    "recentWins": 3,
    "recentPodiums": 4,
    "constructorAvgPos": 1.5,
    "circuitAvgPos": 2.0
  }'
```

**Response**:
```json
{
  "success": true,
  "predictions": {
    "win_probability_percent": 84.7,
    "podium": {"p1": 84.7, "p2": 11.2, "p3": 4.1}
  }
}
```

## Libraries

- pandas (2.0.3) - Data manipulation
- numpy (1.24.3) - Numerical computations
- scikit-learn (1.3.0) - ML models and preprocessing
- xgboost (2.0.3) - Gradient boosting
- flask (3.0.0) - REST API
- flask-cors (4.0.0) - CORS support

## Dataset Source

Ergast F1 Database (1950-2024): https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

## License

MIT

## Author

[Your Name] - CS 5100 Artificial Intelligence - Fall 2024
