/**
 * F1 Race Predictor - ML UI Controller
 * Handles async ML predictions in the UI
 */

class UIControllerML extends UIController {
    constructor(predictor) {
        super(predictor);
        this.isUpdating = false;
    }
    
    /**
     * Update predictions (async for ML model)
     */
    async updatePredictions() {
        if (this.isUpdating) return;  // Prevent concurrent updates
        
        this.isUpdating = true;
        this.showLoadingState();
        
        try {
            // Get predictions (async if using ML)
            const predictions = await this.predictor.predictPodiumWithML();
            const category = this.predictor.getPredictionCategory(predictions.win);
            
            // Update win probability
            document.getElementById('winProbability').textContent = `${predictions.win}%`;
            document.getElementById('winBar').style.width = `${predictions.win}%`;
            
            // Update prediction box styling
            const predictionBox = document.getElementById('predictionBox');
            predictionBox.className = `prediction-box ${category.class}`;
            
            // Update prediction text
            document.getElementById('predictionText').textContent = category.text;
            
            // Update podium probabilities
            this.updatePodiumDisplay(predictions);
            
            this.hideLoadingState();
        } catch (error) {
            console.error('Prediction update failed:', error);
            this.hideLoadingState();
            this.showError('Prediction failed. Using fallback model.');
        }
        
        this.isUpdating = false;
    }
    
    /**
     * Show loading state
     */
    showLoadingState() {
        const predictionBox = document.getElementById('predictionBox');
        if (predictionBox) {
            predictionBox.style.opacity = '0.6';
        }
    }
    
    /**
     * Hide loading state
     */
    hideLoadingState() {
        const predictionBox = document.getElementById('predictionBox');
        if (predictionBox) {
            predictionBox.style.opacity = '1';
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 3000);
    }
}

// Initialize ML UI controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (typeof predictor !== 'undefined') {
        window.uiController = new UIControllerML(predictor);
    }
});