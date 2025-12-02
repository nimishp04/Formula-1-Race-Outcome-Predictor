"""
F1 Race Predictor - Complete ML Model Training
Trains XGBoost, Random Forest, Neural Networks and saves model weights
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  Installing XGBoost...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    HAS_XGBOOST = True

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  matplotlib/seaborn not installed. Skipping plots.")

class F1MLTrainer:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.models_dir = self.project_root / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70 + "\n")
        
    def load_and_engineer_features(self):
        """Load data and create ML features"""
        self.print_header("STEP 1: LOADING DATA & ENGINEERING FEATURES")
        
        data_raw_dir = self.project_root / 'data_raw'
        
        # Load CSVs
        print("Loading CSV files...")
        races = pd.read_csv(data_raw_dir / 'races.csv')
        results = pd.read_csv(data_raw_dir / 'results.csv')
        qualifying = pd.read_csv(data_raw_dir / 'qualifying.csv')
        driver_standings = pd.read_csv(data_raw_dir / 'driver_standings.csv')
        constructor_standings = pd.read_csv(data_raw_dir / 'constructor_standings.csv')
        
        print(f"✓ Loaded {len(races)} races")
        print(f"✓ Loaded {len(results)} results")
        
        # Merge data
        print("\nMerging datasets...")
        merged = results.merge(races[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId')
        merged = merged.merge(
            qualifying[['raceId', 'driverId', 'position']], 
            on=['raceId', 'driverId'],
            how='left',
            suffixes=('', '_qual')
        )
        
        # Sort chronologically
        merged['date'] = pd.to_datetime(merged['date'])
        merged = merged.sort_values(['date', 'raceId'])
        
        print("✓ Data merged successfully")
        
        # Feature Engineering
        print("\nEngineering ML features...")
        featured_data = []
        
        total_rows = len(merged)
        for idx, row in enumerate(merged.iterrows()):
            if idx % 1000 == 0:
                print(f"  Processing: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)", end='\r')
            
            idx, row = row
            
            # Get historical data
            driver_history = merged[
                (merged['driverId'] == row['driverId']) &
                (merged.index < idx)
            ].tail(10)
            
            constructor_history = merged[
                (merged['constructorId'] == row['constructorId']) &
                (merged.index < idx)
            ].tail(10)
            
            circuit_history = merged[
                (merged['driverId'] == row['driverId']) &
                (merged['circuitId'] == row['circuitId']) &
                (merged.index < idx)
            ].tail(5)
            
            if len(driver_history) < 3:  # Need minimum history
                continue
            
            # Calculate features
            features = {
                # Grid & Qualifying
                'grid_position': row['grid'] if row['grid'] > 0 else 20,
                'grid_position_squared': (row['grid'] if row['grid'] > 0 else 20) ** 2,
                
                # Driver Recent Form (Last 5 races)
                'driver_avg_position_5': driver_history.tail(5)['positionOrder'].mean(),
                'driver_best_position_5': driver_history.tail(5)['positionOrder'].min(),
                'driver_worst_position_5': driver_history.tail(5)['positionOrder'].max(),
                'driver_position_std_5': driver_history.tail(5)['positionOrder'].std(),
                
                # Driver Medium Term (Last 10 races)
                'driver_avg_position_10': driver_history['positionOrder'].mean(),
                'driver_wins_10': len(driver_history[driver_history['positionOrder'] == 1]),
                'driver_podiums_10': len(driver_history[driver_history['positionOrder'] <= 3]),
                'driver_top5_10': len(driver_history[driver_history['positionOrder'] <= 5]),
                'driver_points_avg_10': driver_history['points'].mean(),
                
                # Win/Podium rates
                'driver_win_rate': len(driver_history[driver_history['positionOrder'] == 1]) / len(driver_history),
                'driver_podium_rate': len(driver_history[driver_history['positionOrder'] <= 3]) / len(driver_history),
                
                # DNF rate (reliability)
                'driver_dnf_rate': len(driver_history[driver_history['positionOrder'] > 20]) / len(driver_history),
                
                # Constructor Form
                'constructor_avg_position': constructor_history['positionOrder'].mean(),
                'constructor_wins': len(constructor_history[constructor_history['positionOrder'] == 1]),
                'constructor_podiums': len(constructor_history[constructor_history['positionOrder'] <= 3]),
                'constructor_points_avg': constructor_history['points'].mean(),
                
                # Circuit-Specific Performance
                'circuit_avg_position': circuit_history['positionOrder'].mean() if len(circuit_history) > 0 else 10,
                'circuit_best_position': circuit_history['positionOrder'].min() if len(circuit_history) > 0 else 20,
                'circuit_races': len(circuit_history),
                'circuit_wins': len(circuit_history[circuit_history['positionOrder'] == 1]) if len(circuit_history) > 0 else 0,
                
                # Momentum indicators
                'recent_trend': driver_history.tail(3)['positionOrder'].mean() - driver_history.tail(6).head(3)['positionOrder'].mean() if len(driver_history) >= 6 else 0,
                'improving': 1 if (len(driver_history) >= 6 and driver_history.tail(3)['positionOrder'].mean() < driver_history.tail(6).head(3)['positionOrder'].mean()) else 0,
                
                # Consistency
                'consistency_score': 1 / (driver_history.tail(5)['positionOrder'].std() + 1),
                
                # Season context
                'round_number': row['round'],
                'season_progress': row['round'] / 20,  # Normalized
                
                # Target variables
                'won': 1 if row['positionOrder'] == 1 else 0,
                'podium': 1 if row['positionOrder'] <= 3 else 0,
                'top5': 1 if row['positionOrder'] <= 5 else 0,
                'top10': 1 if row['positionOrder'] <= 10 else 0,
                
                # Metadata
                'year': row['year'],
                'raceId': row['raceId']
            }
            
            featured_data.append(features)
        
        print(f"\n✓ Created {len(featured_data)} feature vectors with {len(features)-6} features")
        
        return pd.DataFrame(featured_data)
    
    def prepare_train_test_split(self, df):
        """Prepare train/test split"""
        self.print_header("STEP 2: PREPARING TRAIN/TEST SPLIT")
        
        # Time-based split (important for time series!)
        train_data = df[df['year'] <= 2018]
        test_data = df[df['year'] >= 2019]
        
        print(f"Training data: {len(train_data)} samples (1950-2018)")
        print(f"Test data: {len(test_data)} samples (2019-2024)")
        
        # Features and targets
        feature_cols = [col for col in df.columns if col not in ['won', 'podium', 'top5', 'top10', 'year', 'raceId']]
        self.feature_names = feature_cols
        
        self.X_train = train_data[feature_cols]
        self.X_test = test_data[feature_cols]
        self.y_train = train_data['won']
        self.y_test = test_data['won']
        
        # Also prepare podium targets
        self.y_train_podium = train_data['podium']
        self.y_test_podium = test_data['podium']
        
        print(f"\n✓ Features: {len(feature_cols)}")
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        print(f"✓ Class balance - Wins in training: {self.y_train.sum()} ({self.y_train.mean()*100:.2f}%)")
        
        # Handle missing values (NaN)
        print("\nHandling missing values...")
        nan_counts = self.X_train.isna().sum()
        if nan_counts.sum() > 0:
            print(f"  Found {nan_counts.sum()} NaN values")
            print("  Filling NaN with median values...")
            
            # Fill NaN with median for each column
            from sklearn.impute import SimpleImputer
            self.imputer = SimpleImputer(strategy='median')
            self.X_train = pd.DataFrame(
                self.imputer.fit_transform(self.X_train),
                columns=feature_cols,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.imputer.transform(self.X_test),
                columns=feature_cols,
                index=self.X_test.index
            )
            
            # Save imputer
            with open(self.models_dir / 'imputer.pkl', 'wb') as f:
                pickle.dump(self.imputer, f)
            print("  ✓ NaN values handled")
        else:
            print("  ✓ No NaN values found")
        
        # Standardize features
        print("\nStandardizing features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save scaler
        with open(self.models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✓ Scaler saved")
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "-"*70)
        print("Training XGBoost Classifier...")
        print("-"*70)
        
        # XGBoost parameters (optimized for F1 data)
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42,
            'scale_pos_weight': len(self.y_train) / self.y_train.sum()  # Handle class imbalance
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Save model
        model.save_model(str(self.models_dir / 'xgboost_model.json'))
        with open(self.models_dir / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(self.models_dir / 'xgboost_feature_importance.csv', index=False)
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics
        
        print(f"✓ XGBoost trained - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Top 3 features: {', '.join(feature_importance.head(3)['feature'].tolist())}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "-"*70)
        print("Training Random Forest Classifier...")
        print("-"*70)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Save model
        with open(self.models_dir / 'random_forest_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(self.models_dir / 'random_forest_feature_importance.csv', index=False)
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics
        
        print(f"✓ Random Forest trained - Accuracy: {metrics['accuracy']:.4f}")
        
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "-"*70)
        print("Training Gradient Boosting Classifier...")
        print("-"*70)
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        with open(self.models_dir / 'gradient_boosting_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        self.models['Gradient Boosting'] = model
        self.results['Gradient Boosting'] = metrics
        
        print(f"✓ Gradient Boosting trained - Accuracy: {metrics['accuracy']:.4f}")
        
    def train_neural_network(self):
        """Train Neural Network model"""
        print("\n" + "-"*70)
        print("Training Neural Network (MLP)...")
        print("-"*70)
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        with open(self.models_dir / 'neural_network_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        self.models['Neural Network'] = model
        self.results['Neural Network'] = metrics
        
        print(f"✓ Neural Network trained - Accuracy: {metrics['accuracy']:.4f}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression (baseline)"""
        print("\n" + "-"*70)
        print("Training Logistic Regression (Baseline)...")
        print("-"*70)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        with open(self.models_dir / 'logistic_regression_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = metrics
        
        print(f"✓ Logistic Regression trained - Accuracy: {metrics['accuracy']:.4f}")
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def compare_models(self):
        """Compare all models"""
        self.print_header("STEP 4: MODEL COMPARISON")
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(self.results).T
        comparison = comparison.round(4)
        
        print(comparison.to_string())
        
        # Save comparison
        comparison.to_csv(self.models_dir / 'model_comparison.csv')
        
        # Find best model
        best_model = comparison['accuracy'].idxmax()
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Accuracy: {comparison.loc[best_model, 'accuracy']:.4f}")
        print(f"   F1 Score: {comparison.loc[best_model, 'f1_score']:.4f}")
        
        return comparison
    
    def save_model_info(self, comparison):
        """Save comprehensive model information"""
        self.print_header("STEP 5: SAVING MODEL INFORMATION")
        
        model_info = {
            'training_date': datetime.now().isoformat(),
            'dataset': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'train_period': '1950-2018',
                'test_period': '2019-2024',
                'features': len(self.feature_names),
                'feature_list': self.feature_names
            },
            'models': {
                model_name: {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score']),
                    'roc_auc': float(metrics['roc_auc']),
                    'model_file': f'{model_name.lower().replace(" ", "_")}_model.pkl'
                }
                for model_name, metrics in self.results.items()
            },
            'best_model': comparison['accuracy'].idxmax(),
            'best_accuracy': float(comparison['accuracy'].max())
        }
        
        # Save as JSON
        with open(self.models_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✓ Saved model_info.json")
        
        # Save feature names
        with open(self.models_dir / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print("✓ Saved feature_names.txt")
        
        # Save preprocessing info
        preprocessing_info = {
            'scaler': 'StandardScaler',
            'imputer': 'SimpleImputer(strategy=median)' if self.imputer else 'None',
            'scaler_file': 'scaler.pkl',
            'imputer_file': 'imputer.pkl' if self.imputer else None
        }
        
        with open(self.models_dir / 'preprocessing_info.json', 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        print("✓ Saved preprocessing_info.json")
        
        # Save detailed report
        with open(self.models_dir / 'training_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("F1 RACE PREDICTOR - ML MODEL TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Dataset Information:\n")
            f.write(f"  - Training: {len(self.X_train)} samples (1950-2018)\n")
            f.write(f"  - Testing: {len(self.X_test)} samples (2019-2024)\n")
            f.write(f"  - Features: {len(self.feature_names)}\n\n")
            f.write("Model Comparison:\n")
            f.write(comparison.to_string())
            f.write(f"\n\nBest Model: {comparison['accuracy'].idxmax()}\n")
            f.write(f"Best Accuracy: {comparison['accuracy'].max():.4f}\n")
        
        print("✓ Saved training_report.txt")
        
    def train_all_models(self):
        """Main training pipeline"""
        print("\n" + "="*70)
        print("F1 RACE PREDICTOR - ML MODEL TRAINING")
        print("="*70)
        
        try:
            # Load data
            df = self.load_and_engineer_features()
            
            # Prepare split
            self.prepare_train_test_split(df)
            
            # Train models
            self.print_header("STEP 3: TRAINING MODELS")
            
            # Train each model with error handling
            models_to_train = [
                ('XGBoost', self.train_xgboost),
                ('Random Forest', self.train_random_forest),
                ('Gradient Boosting', self.train_gradient_boosting),
                ('Neural Network', self.train_neural_network),
                ('Logistic Regression', self.train_logistic_regression)
            ]
            
            for model_name, train_func in models_to_train:
                try:
                    train_func()
                except Exception as e:
                    print(f"\n⚠️  Warning: {model_name} training failed: {e}")
                    print(f"   Continuing with other models...")
            
            # Check if we have any successful models
            if not self.models:
                raise Exception("All models failed to train!")
            
            # Compare
            comparison = self.compare_models()
            
            # Save everything
            self.save_model_info(comparison)
            
            # Final summary
            self.print_header("✅ TRAINING COMPLETE!")
            
            print(f"Models saved in: {self.models_dir}")
            print(f"\nSuccessfully trained: {len(self.models)} models")
            print("\nSaved files:")
            for file in sorted(self.models_dir.glob('*')):
                print(f"  ✓ {file.name}")
            
            print("\n" + "="*70)
            print("You can now use these trained models in your application!")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Fatal error during training: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    try:
        trainer = F1MLTrainer()
        trainer.train_all_models()
        
        print("\n✅ SUCCESS! All models trained and saved.")
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")