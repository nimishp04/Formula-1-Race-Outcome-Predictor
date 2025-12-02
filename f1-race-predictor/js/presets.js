/**
 * F1 Race Predictor - Preset Configurations
 * Pre-configured scenarios for quick testing
 */

const PRESETS = {
    champion: {
        name: 'Dominant Champion',
        description: 'Leading championship, pole position, dominant form',
        values: {
            gridPosition: 1,
            qualifyingGap: 0.05,
            driverAvgPos: 1.5,
            recentWins: 4,
            recentPodiums: 5,
            fastestLapRate: 60,
            constructorAvgPos: 1,
            teamMatePosition: 3,
            finishRate: 98,
            championshipPosition: 1,
            pointsGap: 25,
            circuitAvgPos: 1.5,
            circuitType: 'high-speed',
            weatherCondition: 'dry'
        }
    },
    
    wetmaster: {
        name: 'Wet Weather Master',
        description: 'Skilled driver in wet conditions, mid-grid start',
        values: {
            gridPosition: 6,
            qualifyingGap: 0.3,
            driverAvgPos: 4,
            recentWins: 1,
            recentPodiums: 3,
            fastestLapRate: 40,
            constructorAvgPos: 3,
            teamMatePosition: 8,
            finishRate: 92,
            championshipPosition: 3,
            pointsGap: -15,
            circuitAvgPos: 2,
            circuitType: 'street',
            weatherCondition: 'wet'
        }
    },
    
    midfield: {
        name: 'Midfield Charger',
        description: 'Midfield team, consistent driver, lower grid position',
        values: {
            gridPosition: 12,
            qualifyingGap: 0.8,
            driverAvgPos: 8,
            recentWins: 0,
            recentPodiums: 1,
            fastestLapRate: 15,
            constructorAvgPos: 6,
            teamMatePosition: 14,
            finishRate: 88,
            championshipPosition: 8,
            pointsGap: -80,
            circuitAvgPos: 5,
            circuitType: 'technical',
            weatherCondition: 'dry'
        }
    },
    
    contender: {
        name: 'Title Contender',
        description: 'Close championship fight, front of grid',
        values: {
            gridPosition: 3,
            qualifyingGap: 0.15,
            driverAvgPos: 3,
            recentWins: 2,
            recentPodiums: 4,
            fastestLapRate: 45,
            constructorAvgPos: 2,
            teamMatePosition: 5,
            finishRate: 95,
            championshipPosition: 2,
            pointsGap: -5,
            circuitAvgPos: 2,
            circuitType: 'mixed',
            weatherCondition: 'dry'
        }
    }
};

/**
 * Load a preset configuration
 */
function loadPreset(presetName) {
    const preset = PRESETS[presetName];
    
    if (!preset) {
        console.error(`Preset '${presetName}' not found`);
        return;
    }
    
    const inputs = predictor.inputs;
    const values = preset.values;
    
    // Apply all preset values
    inputs.gridPosition.value = values.gridPosition;
    inputs.qualifyingGap.value = values.qualifyingGap;
    inputs.driverAvgPos.value = values.driverAvgPos;
    inputs.recentWins.value = values.recentWins;
    inputs.recentPodiums.value = values.recentPodiums;
    inputs.fastestLapRate.value = values.fastestLapRate;
    inputs.constructorAvgPos.value = values.constructorAvgPos;
    inputs.teamMatePosition.value = values.teamMatePosition;
    inputs.finishRate.value = values.finishRate;
    inputs.championshipPosition.value = values.championshipPosition;
    inputs.pointsGap.value = values.pointsGap;
    inputs.circuitAvgPos.value = values.circuitAvgPos;
    inputs.circuitType.value = values.circuitType;
    inputs.weatherCondition.value = values.weatherCondition;
    
    // Trigger update
    if (window.uiController) {
        window.uiController.update();
    }
    
    // Show notification
    showNotification(`Loaded preset: ${preset.name}`);
}

/**
 * Show notification
 */
function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Export preset to JSON
 */
function exportPreset() {
    const values = predictor.getInputValues();
    const predictions = predictor.predictPodium();
    
    const exportData = {
        timestamp: new Date().toISOString(),
        values: values,
        predictions: predictions
    };
    
    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `f1-prediction-${Date.now()}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
    showNotification('Prediction exported!');
}

/**
 * Import preset from JSON
 */
function importPreset(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            const values = data.values;
            
            // Apply imported values
            Object.keys(values).forEach(key => {
                if (predictor.inputs[key]) {
                    predictor.inputs[key].value = values[key];
                }
            });
            
            // Trigger update
            if (window.uiController) {
                window.uiController.update();
            }
            
            showNotification('Prediction imported!');
        } catch (error) {
            console.error('Failed to import preset:', error);
            showNotification('Import failed!');
        }
    };
    
    reader.readAsText(file);
}