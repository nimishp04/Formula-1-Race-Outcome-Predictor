/**
 * F1 Race Predictor - Core Prediction Engine
 * Neural network-inspired prediction algorithm
 */

class F1Predictor {
    constructor(config = CONFIG) {
        this.config = config;
        this.inputs = this.initializeInputs();
    }
    
    /**
     * Initialize input elements
     */
    initializeInputs() {
        return {
            gridPosition: document.getElementById('gridPosition'),
            qualifyingGap: document.getElementById('qualifyingGap'),
            driverAvgPos: document.getElementById('driverAvgPos'),
            recentWins: document.getElementById('recentWins'),
            recentPodiums: document.getElementById('recentPodiums'),
            fastestLapRate: document.getElementById('fastestLapRate'),
            constructorAvgPos: document.getElementById('constructorAvgPos'),
            teamMatePosition: document.getElementById('teamMatePosition'),
            finishRate: document.getElementById('finishRate'),
            championshipPosition: document.getElementById('championshipPosition'),
            pointsGap: document.getElementById('pointsGap'),
            circuitAvgPos: document.getElementById('circuitAvgPos'),
            circuitType: document.getElementById('circuitType'),
            weatherCondition: document.getElementById('weatherCondition')
        };
    }
    
    /**
     * Get current input values
     */
    getInputValues() {
        return {
            gridPosition: parseInt(this.inputs.gridPosition.value),
            qualifyingGap: parseFloat(this.inputs.qualifyingGap.value),
            driverAvgPos: parseFloat(this.inputs.driverAvgPos.value),
            recentWins: parseInt(this.inputs.recentWins.value),
            recentPodiums: parseInt(this.inputs.recentPodiums.value),
            fastestLapRate: parseInt(this.inputs.fastestLapRate.value),
            constructorAvgPos: parseFloat(this.inputs.constructorAvgPos.value),
            teamMatePosition: parseInt(this.inputs.teamMatePosition.value),
            finishRate: parseInt(this.inputs.finishRate.value),
            championshipPosition: parseInt(this.inputs.championshipPosition.value),
            pointsGap: parseInt(this.inputs.pointsGap.value),
            circuitAvgPos: parseFloat(this.inputs.circuitAvgPos.value),
            circuitType: this.inputs.circuitType.value,
            weatherCondition: this.inputs.weatherCondition.value
        };
    }
    
    /**
     * Layer 1: Core Performance (40%)
     */
    calculateCorePerformance(values) {
        let score = 0;
        
        // Grid position with exponential decay
        const gridWeight = Math.exp(-0.15 * (values.gridPosition - 1));
        score += gridWeight * this.config.WEIGHTS.GRID_POSITION;
        
        // Qualifying gap (performance delta)
        const qualGapScore = Math.max(0, (1 - values.qualifyingGap) * this.config.WEIGHTS.QUALIFYING_GAP);
        score += qualGapScore;
        
        return score;
    }
    
    /**
     * Layer 2: Recent Form & Momentum (30%)
     */
    calculateRecentForm(values) {
        let score = 0;
        
        // Win streak with momentum
        const winMomentum = values.recentWins * 8;
        score += Math.min(winMomentum, 20);
        
        // Podium consistency
        const podiumScore = (values.recentPodiums / 5) * 12;
        score += podiumScore;
        
        // Driver average position with exponential decay
        const driverFormScore = Math.max(0, (10 - values.driverAvgPos) * 1.5);
        score += driverFormScore;
        
        // Fastest lap rate (racecraft indicator)
        score += (values.fastestLapRate / 100) * 8;
        
        return Math.min(score, this.config.WEIGHTS.RECENT_FORM);
    }
    
    /**
     * Layer 3: Team Performance (20%)
     */
    calculateTeamPerformance(values) {
        let score = 0;
        
        // Constructor form
        const constructorScore = Math.max(0, (10 - values.constructorAvgPos) * 2);
        score += constructorScore;
        
        // Teammate comparison (intra-team dominance)
        const teammateGap = Math.abs(values.gridPosition - values.teamMatePosition);
        if (values.gridPosition < values.teamMatePosition) {
            score += Math.min(teammateGap * 2, 8);
        }
        
        // Reliability factor
        score += (values.finishRate / 100) * 5;
        
        return Math.min(score, this.config.WEIGHTS.TEAM_PERFORMANCE);
    }
    
    /**
     * Layer 4: Championship Context & Circuit (10%)
     */
    calculateContextFactors(values) {
        let score = 0;
        
        // Championship position (pressure/motivation)
        if (values.championshipPosition === 1) {
            score += 5;
        } else if (values.championshipPosition <= 3) {
            score += 3;
        }
        
        // Points gap (desperation/comfort)
        if (values.pointsGap > 50) {
            score += 2; // Comfortable lead
        } else if (values.pointsGap < -20) {
            score += 3; // Fighting back
        }
        
        // Circuit expertise
        const circuitScore = Math.max(0, (10 - values.circuitAvgPos) * 0.8);
        score += circuitScore;
        
        // Circuit type bonus
        const circuitBonus = this.getCircuitTypeBonus(values);
        score += circuitBonus;
        
        return Math.min(score, this.config.WEIGHTS.CIRCUIT_CONTEXT);
    }
    
    /**
     * Get circuit type specific bonus
     */
    getCircuitTypeBonus(values) {
        let bonus = 0;
        
        switch(values.circuitType) {
            case 'street':
                if (values.gridPosition === 1) {
                    bonus += this.config.CIRCUIT_BONUSES.street.polePosition;
                } else if (values.gridPosition <= 3) {
                    bonus += this.config.CIRCUIT_BONUSES.street.top3Grid;
                }
                break;
            case 'high-speed':
                if (values.constructorAvgPos <= 3) {
                    bonus += this.config.CIRCUIT_BONUSES.highSpeed.constructorBonus;
                }
                break;
            case 'technical':
                if (values.circuitAvgPos <= 3) {
                    bonus += this.config.CIRCUIT_BONUSES.technical.circuitKnowledge;
                }
                break;
        }
        
        return bonus;
    }
    
    /**
     * Apply weather adjustment
     */
    applyWeatherMultiplier(baseScore, values) {
        const weather = values.weatherCondition;
        
        if (weather === 'dry') {
            return baseScore;
        }
        
        const multipliers = this.config.WEATHER_MULTIPLIERS[weather];
        
        // Wet/mixed conditions benefit skilled, reliable drivers
        if (values.driverAvgPos <= 5 && values.finishRate >= 90) {
            return baseScore * multipliers.skilled;
        }
        
        // Pole position less valuable in wet
        if (values.gridPosition <= 3) {
            return baseScore * multipliers.polePosition;
        }
        
        return baseScore;
    }
    
    /**
     * Main prediction function
     */
    predictWinProbability() {
        const values = this.getInputValues();
        
        // Calculate all layers
        const layer1 = this.calculateCorePerformance(values);
        const layer2 = this.calculateRecentForm(values);
        const layer3 = this.calculateTeamPerformance(values);
        const layer4 = this.calculateContextFactors(values);
        
        // Sum all layers
        let baseScore = layer1 + layer2 + layer3 + layer4;
        
        // Apply weather adjustment
        const adjustedScore = this.applyWeatherMultiplier(baseScore, values);
        
        // Normalize to 0-100 range
        return Math.min(Math.round(adjustedScore), 100);
    }
    
    /**
     * Predict podium positions
     */
    predictPodium() {
        const winProb = this.predictWinProbability();
        const values = this.getInputValues();
        
        let p2Prob = 0;
        let p3Prob = 0;
        
        // Calculate P2 and P3 probabilities based on grid position
        if (values.gridPosition <= 6) {
            p2Prob = Math.max(0, 35 - values.gridPosition * 4 + (values.recentPodiums * 3));
            p3Prob = Math.max(0, 30 - values.gridPosition * 3 + (values.recentPodiums * 2));
        } else if (values.gridPosition <= 10) {
            p2Prob = Math.max(0, 20 - values.gridPosition * 2);
            p3Prob = Math.max(0, 25 - values.gridPosition * 2);
        } else {
            p2Prob = Math.max(0, 10 - values.gridPosition);
            p3Prob = Math.max(0, 15 - values.gridPosition);
        }
        
        // Apply circuit type modifier
        if (values.circuitType === 'street') {
            // Street circuits: qualifying more important
            p2Prob *= 0.8;
            p3Prob *= 0.7;
        }
        
        const totalPodiumProb = Math.min(winProb + p2Prob + p3Prob, 95);
        
        return {
            win: winProb,
            p2: Math.min(Math.round(p2Prob), 40),
            p3: Math.min(Math.round(p3Prob), 35),
            anyPodium: Math.round(totalPodiumProb)
        };
    }
    
    /**
     * Get prediction category
     */
    getPredictionCategory(winProb) {
        const thresholds = this.config.THRESHOLDS;
        
        if (winProb >= thresholds.strongFavorite) {
            return {
                class: 'high',
                text: '🏆 STRONG WIN CANDIDATE',
                color: 'prediction-high'
            };
        } else if (winProb >= thresholds.competitive) {
            return {
                class: 'medium',
                text: '⚡ COMPETITIVE CONTENDER',
                color: 'prediction-medium'
            };
        } else if (winProb >= thresholds.possible) {
            return {
                class: 'low',
                text: '📊 Possible Contender',
                color: 'prediction-low'
            };
        } else {
            return {
                class: 'very-low',
                text: '🎯 Outsider',
                color: 'prediction-very-low'
            };
        }
    }
}

// Initialize predictor
const predictor = new F1Predictor();