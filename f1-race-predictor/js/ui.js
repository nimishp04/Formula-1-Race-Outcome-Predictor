/**
 * F1 Race Predictor - UI Controller
 * Handles all UI updates and user interactions
 */

class UIController {
    constructor(predictor) {
        this.predictor = predictor;
        this.updateTimer = null;
        this.initializeEventListeners();
        this.update(); // Initial update
    }
    
    /**
     * Initialize all event listeners
     */
    initializeEventListeners() {
        const inputs = this.predictor.inputs;
        
        // Add change listeners to all inputs
        Object.values(inputs).forEach(input => {
            input.addEventListener('input', () => this.debouncedUpdate());
            input.addEventListener('change', () => this.debouncedUpdate());
        });
    }
    
    /**
     * Debounced update to prevent excessive calculations
     */
    debouncedUpdate() {
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.update(), CONFIG.UPDATE_DELAY);
    }
    
    /**
     * Update all displays
     */
    update() {
        this.updateInputDisplays();
        this.updatePredictions();
    }
    
    /**
     * Update input display values
     */
    updateInputDisplays() {
        const values = this.predictor.getInputValues();
        
        // Update all display elements
        document.getElementById('gridDisplay').textContent = `P${values.gridPosition}`;
        document.getElementById('qualGapDisplay').textContent = `${values.qualifyingGap.toFixed(2)}s`;
        document.getElementById('driverAvgDisplay').textContent = values.driverAvgPos.toFixed(1);
        document.getElementById('winsDisplay').textContent = values.recentWins;
        document.getElementById('podiumsDisplay').textContent = values.recentPodiums;
        document.getElementById('fastestLapDisplay').textContent = `${values.fastestLapRate}%`;
        document.getElementById('constructorDisplay').textContent = values.constructorAvgPos.toFixed(1);
        document.getElementById('teammateDisplay').textContent = `P${values.teamMatePosition}`;
        document.getElementById('finishDisplay').textContent = `${values.finishRate}%`;
        document.getElementById('champDisplay').textContent = `P${values.championshipPosition}`;
        document.getElementById('pointsGapDisplay').textContent = 
            values.pointsGap > 0 ? `+${values.pointsGap} pts` : `${values.pointsGap} pts`;
        document.getElementById('circuitDisplay').textContent = values.circuitAvgPos.toFixed(1);
    }
    
    /**
     * Update predictions display
     */
    updatePredictions() {
        const predictions = this.predictor.predictPodium();
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
    }
    
    /**
     * Update podium display
     */
    updatePodiumDisplay(predictions) {
        // P1
        document.getElementById('p1Prob').textContent = `${predictions.win}%`;
        document.getElementById('p1Bar').style.width = `${predictions.win}%`;
        
        // P2
        document.getElementById('p2Prob').textContent = `${predictions.p2}%`;
        document.getElementById('p2Bar').style.width = `${predictions.p2}%`;
        
        // P3
        document.getElementById('p3Prob').textContent = `${predictions.p3}%`;
        document.getElementById('p3Bar').style.width = `${predictions.p3}%`;
        
        // Any podium
        document.getElementById('anyPodiumProb').textContent = `${predictions.anyPodium}%`;
    }
    
    /**
     * Show loading state
     */
    showLoading() {
        document.getElementById('predictionBox').classList.add('loading');
    }
    
    /**
     * Hide loading state
     */
    hideLoading() {
        document.getElementById('predictionBox').classList.remove('loading');
    }
}

// Initialize UI controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const uiController = new UIController(predictor);
});