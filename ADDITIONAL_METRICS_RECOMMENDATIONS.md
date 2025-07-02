# Additional Performance Metrics for COE Prediction Models

## Current Comprehensive Framework

### ✅ Implemented Metrics (Excellent Foundation)
1. **Accuracy**: MAPE, R²
2. **Directional Reliability**: Direction Accuracy, Hit Rate
3. **Risk Awareness**: Volatility Correlation, Max Drawdown  
4. **Economic Viability**: MAE, RMSE, Sharpe Ratio, Information Ratio, Calmar Ratio

## Recommended Additional Metrics

### 1. **Advanced Risk Metrics**

#### Value at Risk (VaR)
```python
def calculate_var(returns, confidence_level=0.05):
    """Calculate Value at Risk at given confidence level"""
    return np.percentile(returns, confidence_level * 100)
```
**Why Important**: Shows worst-case loss expectations
**Industry Standard**: 95% and 99% confidence levels
**Business Value**: Risk budgeting and capital allocation

#### Conditional Value at Risk (CVaR)
```python
def calculate_cvar(returns, confidence_level=0.05):
    """Calculate Conditional VaR (Expected Shortfall)"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()
```
**Why Important**: Average loss beyond VaR threshold
**Regulatory Requirement**: Basel III compliance for financial institutions

### 2. **Market Timing Metrics**

#### Timing Ratio
```python
def calculate_timing_ratio(predictions, actual_returns):
    """Measure market timing ability"""
    correct_up_calls = sum((pred > 0.5) & (ret > 0) for pred, ret in zip(predictions, actual_returns))
    total_up_calls = sum(pred > 0.5 for pred in predictions)
    return correct_up_calls / total_up_calls if total_up_calls > 0 else 0
```
**Why Important**: Measures timing skill separately from selection skill

#### Market Regime Detection
```python
def calculate_regime_consistency(predictions, market_volatility):
    """Performance consistency across market regimes"""
    high_vol_mask = market_volatility > np.median(market_volatility)
    low_vol_performance = calculate_accuracy(predictions[~high_vol_mask])
    high_vol_performance = calculate_accuracy(predictions[high_vol_mask])
    return abs(high_vol_performance - low_vol_performance)
```
**Why Important**: Shows model robustness across market conditions

### 3. **Business Impact Metrics**

#### Profit Factor
```python
def calculate_profit_factor(strategy_returns):
    """Ratio of gross profit to gross loss"""
    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')
```
**Why Important**: Direct measure of trading strategy profitability
**Industry Benchmark**: >1.5 is good, >2.0 is excellent

#### Win Rate vs Average Win/Loss
```python
def calculate_win_loss_ratio(strategy_returns):
    """Analyze win rate and average win/loss amounts"""
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    
    win_rate = len(wins) / len(strategy_returns)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf')
    }
```

### 4. **Model Stability Metrics**

#### Prediction Stability
```python
def calculate_prediction_stability(model, X_test, n_bootstrap=100):
    """Measure prediction consistency across bootstrap samples"""
    predictions = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_bootstrap = X_test.iloc[indices]
        pred = model.predict(X_bootstrap)
        predictions.append(pred)
    
    # Calculate prediction variance
    prediction_std = np.std(predictions, axis=0)
    return np.mean(prediction_std)
```
**Why Important**: Measures model reliability and confidence

#### Feature Importance Stability
```python
def calculate_feature_stability(model, X_train, y_train, n_iterations=50):
    """Measure feature importance consistency"""
    feature_importances = []
    for _ in range(n_iterations):
        # Bootstrap training
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]
        
        model.fit(X_boot, y_boot)
        importance = model.feature_importances_
        feature_importances.append(importance)
    
    # Calculate stability
    importance_std = np.std(feature_importances, axis=0)
    importance_mean = np.mean(feature_importances, axis=0)
    stability = 1 - (importance_std / importance_mean).mean()
    return stability
```

### 5. **Regulatory Compliance Metrics**

#### Model Interpretability Score
```python
def calculate_interpretability_score(model, feature_names):
    """Quantify model interpretability"""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        top_features = np.argsort(model.feature_importances_)[-5:]
        importance_concentration = model.feature_importances_[top_features].sum()
        return importance_concentration
    else:
        # Linear models
        if hasattr(model, 'coef_'):
            return np.sum(np.abs(model.coef_)) / len(model.coef_)
    return 0.5  # Default for black-box models
```

#### Fairness Across Categories
```python
def calculate_category_fairness(predictions, actual_values, categories):
    """Ensure fair performance across COE categories"""
    category_metrics = {}
    for category in np.unique(categories):
        mask = categories == category
        cat_pred = predictions[mask]
        cat_actual = actual_values[mask]
        
        category_metrics[category] = {
            'mape': calculate_mape(cat_actual, cat_pred),
            'direction_accuracy': calculate_direction_accuracy(cat_actual, cat_pred)
        }
    
    # Calculate fairness as consistency across categories
    mape_values = [m['mape'] for m in category_metrics.values()]
    mape_fairness = 1 - (np.std(mape_values) / np.mean(mape_values))
    
    return mape_fairness, category_metrics
```

## Priority Implementation Order

### Phase 1: Immediate Value (High Impact, Low Effort)
1. **Profit Factor**: Direct business impact measurement
2. **Win Rate Analysis**: Trading strategy viability
3. **VaR/CVaR**: Risk management compliance

### Phase 2: Advanced Analytics (Medium Effort, High Academic Value)
1. **Timing Ratio**: Market timing skill assessment
2. **Regime Consistency**: Model robustness validation
3. **Prediction Stability**: Model reliability quantification

### Phase 3: Research Excellence (High Effort, Maximum Academic Impact)
1. **Feature Stability**: Advanced model validation
2. **Interpretability Scoring**: Regulatory compliance
3. **Category Fairness**: Ethical AI demonstration

## Academic Presentation Benefits

### Demonstrates Graduate-Level Understanding
- **Risk Management**: VaR/CVaR shows institutional knowledge
- **Business Acumen**: Profit Factor connects to real trading
- **Model Validation**: Stability metrics show rigorous evaluation
- **Ethical AI**: Fairness metrics demonstrate responsible development

### Industry Readiness
- **Regulatory Awareness**: Compliance metrics
- **Practical Application**: Business impact measures
- **Professional Standards**: Institutional-grade evaluation

## Implementation for COE Project

```python
def comprehensive_model_evaluation(model, predictions, actual_values, strategy_returns):
    """Complete evaluation suite for COE models"""
    
    results = {
        # Existing metrics
        'accuracy_metrics': calculate_accuracy_metrics(predictions, actual_values),
        'risk_metrics': calculate_risk_metrics(strategy_returns),
        'economic_metrics': calculate_economic_metrics(strategy_returns),
        
        # New advanced metrics
        'advanced_risk': {
            'var_95': calculate_var(strategy_returns, 0.05),
            'cvar_95': calculate_cvar(strategy_returns, 0.05),
            'var_99': calculate_var(strategy_returns, 0.01)
        },
        'business_impact': {
            'profit_factor': calculate_profit_factor(strategy_returns),
            'win_loss_analysis': calculate_win_loss_ratio(strategy_returns)
        },
        'model_stability': {
            'prediction_stability': calculate_prediction_stability(model, X_test),
            'feature_stability': calculate_feature_stability(model, X_train, y_train)
        }
    }
    
    return results
```

These additional metrics would elevate your project to PhD-level sophistication while maintaining practical business relevance.