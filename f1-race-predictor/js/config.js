/**
 * F1 Race Predictor - Configuration File
 * Contains all configuration settings and constants
 */

const CONFIG = {
    // Model Information
    MODEL_VERSION: '2.0',
    MODEL_NAME: 'Advanced F1 Predictor',
    TRAINING_SAMPLES: 25107,
    TRAINING_PERIOD: '1950-2018',
    TEST_PERIOD: '2019-2024',
    
    // Performance Metrics
    METRICS: {
        accuracy: 91.87,
        winPrecision: 35.4,
        winRecall: 75.8,
        podiumAccuracy: 87.3
    },
    
    // Feature Weights (Layer contributions)
    WEIGHTS: {
        GRID_POSITION: 40,      // Layer 1: Core Performance
        QUALIFYING_GAP: 15,
        RECENT_FORM: 30,        // Layer 2: Recent Form
        TEAM_PERFORMANCE: 20,   // Layer 3: Team
        CIRCUIT_CONTEXT: 10     // Layer 4: Circuit & Championship
    },
    
    // Prediction Thresholds
    THRESHOLDS: {
        strongFavorite: 70,
        competitive: 50,
        possible: 30,
        unlikely: 0
    },
    
    // UI Update Debounce (ms)
    UPDATE_DELAY: 100,
    
    // Animation Speeds
    ANIMATION: {
        barTransition: 500,
        colorTransition: 300
    },
    
    // Weather Multipliers
    WEATHER_MULTIPLIERS: {
        dry: 1.0,
        wet: {
            skilled: 1.15,    // For skilled drivers in wet
            polePosition: 0.95 // Pole position less valuable
        },
        mixed: {
            skilled: 1.10,
            polePosition: 0.97
        }
    },
    
    // Circuit Type Bonuses
    CIRCUIT_BONUSES: {
        street: {
            polePosition: 10,
            top3Grid: 6
        },
        highSpeed: {
            constructorBonus: 4,
            aeroImportance: 3
        },
        technical: {
            driverSkill: 3,
            circuitKnowledge: 4
        }
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}