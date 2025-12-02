/**
 * F1 Race Predictor - ML Model Integration
 * Calls trained Neural Network model via API
 */

class F1PredictorML extends F1Predictor {
    constructor(config = CONFIG) {
        super(config);
        this.apiUrl = 'http://localhost:5000';
        this.useMLModel = true;  // Toggle between ML and rule-based
        this.apiAvailable = false;
        this.checkAPIHealth();
    }
    
    /**
     * Check if API is available
     */
    async checkAPIHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            this.apiAvailable = data.status === 'healthy' && data.model_loaded;
            
            if (this.apiAvailable) {
                console.log('✓ ML Model API connected');
                this.showMLStatus(true);
            } else {
                console.warn('⚠️ ML Model API unavailable, using rule-based fallback');
                this.showMLStatus(false);
            }
        } catch (error) {
            console.warn('⚠️ Cannot connect to ML API, using rule-based fallback');
            this.apiAvailable = false;
            this.showMLStatus(false);
        }
    }
    
    /**
     * Show ML model status in UI
     */
    showMLStatus(available) {
        const statusElement = document.getElementById('mlStatus');
        if (statusElement) {
            if (available) {
                statusElement.innerHTML = '<span style="color: #10b981;">● ML Model Active (95.54% accuracy)</span>';
                statusElement.style.display = 'block';
            } else {
                statusElement.innerHTML = '<span style="color: #f59e0b;">● Using Rule-Based Model (91.87% accuracy)</span>';
                statusElement.style.display = 'block';
            }
        }
    }
    
    /**
     * Make prediction using ML model via API
     */
    async predictWithMLModel() {
        const values = this.getInputValues();
        
        // Prepare API request
        const requestData = {
            gridPosition: values.gridPosition,
            driverAvgPos: values.driverAvgPos,
            recentWins: values.recentWins,
            recentPodiums: values.recentPodiums,
            constructorAvgPos: values.constructorAvgPos,
            circuitAvgPos: values.circuitAvgPos,
            fastestLapRate: values.fastestLapRate,
            finishRate: values.finishRate,
            roundNumber: 10  // Default mid-season
        };
        
        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`API returned ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                return data.predictions.win_probability_percent;
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('ML prediction failed:', error);
            // Fallback to rule-based
            return this.predictWinProbability();
        }
    }
    
    /**
     * Predict podium using ML model
     */
    async predictPodiumWithML() {
        if (!this.useMLModel || !this.apiAvailable) {
            // Use rule-based fallback
            return this.predictPodium();
        }
        
        const values = this.getInputValues();
        
        const requestData = {
            gridPosition: values.gridPosition,
            driverAvgPos: values.driverAvgPos,
            recentWins: values.recentWins,
            recentPodiums: values.recentPodiums,
            constructorAvgPos: values.constructorAvgPos,
            circuitAvgPos: values.circuitAvgPos,
            fastestLapRate: values.fastestLapRate,
            finishRate: values.finishRate,
            roundNumber: 10
        };
        
        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            const data = await response.json();
            
            if (data.success && data.predictions.podium) {
                return {
                    win: Math.round(data.predictions.podium.p1),
                    p2: Math.round(data.predictions.podium.p2),
                    p3: Math.round(data.predictions.podium.p3),
                    anyPodium: Math.round(data.predictions.podium.any_podium)
                };
            } else {
                throw new Error('Podium prediction failed');
            }
        } catch (error) {
            console.error('ML podium prediction failed:', error);
            // Fallback to rule-based
            return this.predictPodium();
        }
    }
    
    /**
     * Toggle between ML and rule-based prediction
     */
    togglePredictionMode() {
        this.useMLModel = !this.useMLModel;
        console.log(`Prediction mode: ${this.useMLModel ? 'ML Model' : 'Rule-Based'}`);
        
        // Update UI
        const button = document.getElementById('toggleModelBtn');
        if (button) {
            button.textContent = this.useMLModel ? '🤖 Using ML Model' : '📊 Using Rule-Based';
        }
    }
}

// Override the global predictor with ML version
if (typeof predictor !== 'undefined') {
    predictor = new F1PredictorML();
    console.log('F1 Predictor upgraded to ML mode');
}