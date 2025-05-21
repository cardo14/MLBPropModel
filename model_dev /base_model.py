# MLB Player Prop Betting Models
# Focus: Walks and Strikeouts prediction with XGBoost and Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys

# Add the model dev directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from data_preprocessor
try:
    # Import standard preprocessed data
    from data_preprocessor import X_walks_train, X_walks_test, y_walks_train, y_walks_test
    from data_preprocessor import X_ks_train, X_ks_test, y_ks_train, y_ks_test
    
    # Import PCA preprocessed data
    from data_preprocessor import X_walks_train_pca, X_walks_test_pca, y_walks_train_pca, y_walks_test_pca
    from data_preprocessor import X_ks_train_pca, X_ks_test_pca, y_ks_train_pca, y_ks_test_pca
    
    data_loaded = True
    pca_data_loaded = True
except ImportError as e:
    print(f"Could not import preprocessed data: {e}. Using dummy data for demonstration.")
    # Create dummy data for demonstration
    X_walks_train = np.random.rand(100, 10)
    X_walks_test = np.random.rand(20, 10)
    y_walks_train = np.random.randint(0, 5, 100)
    y_walks_test = np.random.randint(0, 5, 20)
    
    X_ks_train = np.random.rand(100, 10)
    X_ks_test = np.random.rand(20, 10)
    y_ks_train = np.random.randint(0, 10, 100)
    y_ks_test = np.random.randint(0, 10, 20)
    
    # Create dummy PCA data with fewer dimensions
    X_walks_train_pca = np.random.rand(100, 5)
    X_walks_test_pca = np.random.rand(20, 5)
    y_walks_train_pca = y_walks_train.copy()
    y_walks_test_pca = y_walks_test.copy()
    
    X_ks_train_pca = np.random.rand(100, 5)
    X_ks_test_pca = np.random.rand(20, 5)
    y_ks_train_pca = y_ks_train.copy()
    y_ks_test_pca = y_ks_test.copy()
    
    data_loaded = False
    pca_data_loaded = False

# Enhanced evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """Evaluate a model's performance on training and test data with enhanced metrics"""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Round predictions to nearest 0.5 for better evaluation
    # This makes sense for walks/strikeouts which are discrete values
    test_preds_rounded = np.round(test_preds * 2) / 2
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_rmse_rounded = np.sqrt(mean_squared_error(y_test, test_preds_rounded))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    # Calculate accuracy within 0.5 units (important for betting)
    within_half_unit = np.mean(np.abs(test_preds - y_test) <= 0.5)
    
    # Calculate accuracy for over/under predictions at common lines
    common_lines = [0.5, 1.5, 2.5]
    line_accuracies = {}
    for line in common_lines:
        actual_over = y_test > line
        pred_over = test_preds > line
        accuracy = np.mean(actual_over == pred_over)
        line_accuracies[line] = accuracy
    
    print(f"\n{model_name} Results:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test RMSE (Rounded): {test_rmse_rounded:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Accuracy within 0.5 units: {within_half_unit:.4f}")
    
    print("Over/Under Prediction Accuracy:")
    for line, acc in line_accuracies.items():
        print(f"  Line {line}: {acc:.4f}")
    
    return model, test_preds, test_rmse

# Function to perform bootstrap simulation for probability distributions
def simulate_predictions(model, X, n_simulations=1000, noise_level=0.1):
    """Generate simulated predictions to create probability distributions"""
    # Get base prediction
    base_prediction = model.predict(X)
    
    # Create array to store simulations
    simulations = np.zeros((len(X), n_simulations))
    
    # Generate simulations with noise
    for i in range(n_simulations):
        # Add random noise based on the model's expected error
        noise = np.random.normal(0, noise_level, size=len(X))
        simulations[:, i] = base_prediction + noise
        
        # Ensure non-negative values for walks/strikeouts
        simulations[:, i] = np.maximum(0, simulations[:, i])
    
    return simulations

# Function to calculate over/under probabilities
def calculate_over_under_probs(simulations, line):
    """Calculate probability of going over/under a given line"""
    # Ensure simulations is a numpy array
    if not isinstance(simulations, np.ndarray):
        simulations = np.array(simulations)
    
    # Handle empty simulations
    if simulations.size == 0:
        return np.array([]), np.array([])
    
    # Ensure simulations has the right shape
    if len(simulations.shape) == 1:
        simulations = simulations.reshape(1, -1)
    
    # Count simulations over the line
    over_count = np.sum(simulations > line, axis=1)
    
    # Calculate probabilities
    prob_over = over_count / simulations.shape[1]
    prob_under = 1 - prob_over
    
    return prob_over, prob_under

# Function to calculate expected value
def calculate_ev(prob, american_odds):
    """Calculate expected value of a bet given probability and American odds"""
    if american_odds > 0:
        decimal_odds = (american_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(american_odds)) + 1
    
    return (prob * (decimal_odds - 1)) - (1 - prob)

# Function to find value bets
def find_value_bets(simulations, lines, over_odds, under_odds, threshold=0.05):
    """Find value bets based on simulation results"""
    value_bets = []
    
    for i in range(len(lines)):
        line = lines[i]
        prob_over, prob_under = calculate_over_under_probs(simulations, line)
        
        # Check if probabilities exist
        if len(prob_over) == 0 or len(prob_under) == 0:
            print(f"Warning: No probability data for line {line}")
            continue
        
        # Calculate EV for over and under
        ev_over = calculate_ev(prob_over[0], over_odds[i])
        ev_under = calculate_ev(prob_under[0], under_odds[i])
        
        # Check if either bet has value
        if ev_over > threshold:
            value_bets.append({
                'index': i,
                'line': line,
                'bet': 'OVER',
                'prob': prob_over[0],
                'odds': over_odds[i],
                'ev': ev_over
            })
        
        if ev_under > threshold:
            value_bets.append({
                'index': i,
                'line': line,
                'bet': 'UNDER',
                'prob': prob_under[0],
                'odds': under_odds[i],
                'ev': ev_under
            })
    
    return value_bets

# Create output directory for models
os.makedirs('models', exist_ok=True)

# WALKS MODELS
print("\n" + "=" * 50)
print("TRAINING WALKS MODELS (STANDARD FEATURES)")
print("=" * 50)

# Random Forest for walks with optimized hyperparameters
rf_walks = RandomForestRegressor(
    n_estimators=200,       # Increased from 100 for better ensemble learning
    max_depth=6,            # Slightly increased to capture more complex patterns
    min_samples_split=6,    # Increased to prevent overfitting
    min_samples_leaf=3,     # Increased to prevent overfitting
    max_features='sqrt',    # Use only a subset of features for each split
    bootstrap=True,         # Use bootstrapping for better generalization
    oob_score=True,         # Use out-of-bag samples to estimate performance
    n_jobs=-1,
    random_state=42
)

# Use cross-validation to get more reliable performance metrics
rf_walks_cv_scores = cross_val_score(rf_walks, X_walks_train, y_walks_train, 
                                    cv=5, scoring='neg_mean_squared_error')
print(f"RF Walks CV RMSE: {np.sqrt(-rf_walks_cv_scores.mean()):.4f} ± {np.sqrt(-rf_walks_cv_scores).std():.4f}")

# Train and evaluate the model
rf_walks_model, rf_walks_preds, rf_walks_rmse = evaluate_model(
    rf_walks, X_walks_train, X_walks_test, y_walks_train, y_walks_test, "Random Forest - Walks"
)
joblib.dump(rf_walks_model, 'models/rf_walks_model.joblib')

# Analyze feature importance for walks model
if data_loaded:
    print("\nFeature Importance - Random Forest Walks Model:")
    feature_names = X_walks_train.columns if hasattr(X_walks_train, 'columns') else [f"Feature {i}" for i in range(X_walks_train.shape[1])]
    importances = rf_walks_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print top 10 features
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# XGBoost for walks with optimized hyperparameters
xgb_walks = xgb.XGBRegressor(
    n_estimators=150,       # Increased for better ensemble learning
    max_depth=5,            # Slightly increased to capture more patterns
    learning_rate=0.03,     # Reduced for better generalization
    subsample=0.8,          # Use 80% of samples for each tree
    colsample_bytree=0.8,   # Use 80% of features for each tree
    colsample_bylevel=0.7,  # Use 70% of features at each level
    reg_alpha=0.8,          # L1 regularization
    reg_lambda=1.2,         # L2 regularization
    gamma=1,                # Minimum loss reduction for split
    min_child_weight=3,     # Minimum sum of instance weight in a child
    objective='reg:squarederror',
    random_state=42
)

# Use cross-validation for XGBoost as well
xgb_walks_cv_scores = cross_val_score(xgb_walks, X_walks_train, y_walks_train, 
                                     cv=5, scoring='neg_mean_squared_error')
print(f"XGB Walks CV RMSE: {np.sqrt(-xgb_walks_cv_scores.mean()):.4f} ± {np.sqrt(-xgb_walks_cv_scores).std():.4f}")
xgb_walks_model, xgb_walks_preds, xgb_walks_rmse = evaluate_model(
    xgb_walks, X_walks_train, X_walks_test, y_walks_train, y_walks_test, "XGBoost - Walks"
)
joblib.dump(xgb_walks_model, 'models/xgb_walks_model.joblib')

# Analyze feature importance for XGBoost walks model
if data_loaded:
    print("\nFeature Importance - XGBoost Walks Model:")
    feature_names = X_walks_train.columns if hasattr(X_walks_train, 'columns') else [f"Feature {i}" for i in range(X_walks_train.shape[1])]
    importances = xgb_walks_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print top 10 features
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# STRIKEOUTS MODELS
print("\n" + "=" * 50)
print("TRAINING STRIKEOUTS MODELS (STANDARD FEATURES)")
print("=" * 50)

# Random Forest for strikeouts with optimized hyperparameters
rf_ks = RandomForestRegressor(
    n_estimators=200,       # Increased from 100 for better ensemble learning
    max_depth=7,            # Increased to capture more complex strikeout patterns
    min_samples_split=5,    # Keep the same to prevent overfitting
    min_samples_leaf=2,     # Keep the same to prevent overfitting
    max_features='sqrt',    # Use only a subset of features for each split
    bootstrap=True,         # Use bootstrapping for better generalization
    oob_score=True,         # Use out-of-bag samples to estimate performance
    n_jobs=-1,
    random_state=42
)

# Use cross-validation for strikeouts model
rf_ks_cv_scores = cross_val_score(rf_ks, X_ks_train, y_ks_train, 
                                 cv=5, scoring='neg_mean_squared_error')
print(f"RF Strikeouts CV RMSE: {np.sqrt(-rf_ks_cv_scores.mean()):.4f} ± {np.sqrt(-rf_ks_cv_scores).std():.4f}")

# Train and evaluate the model
rf_ks_model, rf_ks_preds, rf_ks_rmse = evaluate_model(
    rf_ks, X_ks_train, X_ks_test, y_ks_train, y_ks_test, "Random Forest - Strikeouts"
)
joblib.dump(rf_ks_model, 'models/rf_strikeouts_model.joblib')

# Analyze feature importance for strikeouts model
if data_loaded:
    print("\nFeature Importance - Random Forest Strikeouts Model:")
    feature_names = X_ks_train.columns if hasattr(X_ks_train, 'columns') else [f"Feature {i}" for i in range(X_ks_train.shape[1])]
    importances = rf_ks_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print top 10 features
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# XGBoost for strikeouts with optimized hyperparameters
xgb_ks = xgb.XGBRegressor(
    n_estimators=150,       # Increased for better ensemble learning
    max_depth=6,            # Increased to capture more complex strikeout patterns
    learning_rate=0.03,     # Reduced for better generalization
    subsample=0.8,          # Use 80% of samples for each tree
    colsample_bytree=0.8,   # Use 80% of features for each tree
    colsample_bylevel=0.7,  # Use 70% of features at each level
    reg_alpha=0.5,          # L1 regularization
    reg_lambda=1.5,         # L2 regularization - increased for strikeouts
    gamma=0.5,              # Minimum loss reduction for split
    min_child_weight=3,     # Minimum sum of instance weight in a child
    objective='reg:squarederror',
    random_state=42
)

# Use cross-validation for XGBoost strikeouts model
xgb_ks_cv_scores = cross_val_score(xgb_ks, X_ks_train, y_ks_train, 
                                  cv=5, scoring='neg_mean_squared_error')
print(f"XGB Strikeouts CV RMSE: {np.sqrt(-xgb_ks_cv_scores.mean()):.4f} ± {np.sqrt(-xgb_ks_cv_scores).std():.4f}")
xgb_ks_model, xgb_ks_preds, xgb_ks_rmse = evaluate_model(
    xgb_ks, X_ks_train, X_ks_test, y_ks_train, y_ks_test, "XGBoost - Strikeouts"
)
joblib.dump(xgb_ks_model, 'models/xgb_strikeouts_model.joblib')

# WALKS MODELS WITH PCA
if pca_data_loaded:
    print("\n" + "=" * 50)
    print("TRAINING WALKS MODELS WITH PCA")
    print("=" * 50)
    
    # Random Forest for walks with PCA
    rf_walks_pca = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Use cross-validation to get more reliable performance metrics
    rf_walks_pca_cv_scores = cross_val_score(rf_walks_pca, X_walks_train_pca, y_walks_train_pca, 
                                        cv=5, scoring='neg_mean_squared_error')
    print(f"RF Walks PCA CV RMSE: {np.sqrt(-rf_walks_pca_cv_scores.mean()):.4f} ± {np.sqrt(-rf_walks_pca_cv_scores).std():.4f}")
    
    # Train and evaluate the model
    rf_walks_pca_model, rf_walks_pca_preds, rf_walks_pca_rmse = evaluate_model(
        rf_walks_pca, X_walks_train_pca, X_walks_test_pca, y_walks_train_pca, y_walks_test_pca, "Random Forest - Walks (PCA)"
    )
    joblib.dump(rf_walks_pca_model, 'models/rf_walks_pca_model.joblib')
    
    # XGBoost for walks with PCA
    xgb_walks_pca = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        reg_alpha=0.8,
        reg_lambda=1.2,
        gamma=1,
        min_child_weight=3,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Use cross-validation for XGBoost as well
    xgb_walks_pca_cv_scores = cross_val_score(xgb_walks_pca, X_walks_train_pca, y_walks_train_pca, 
                                        cv=5, scoring='neg_mean_squared_error')
    print(f"XGB Walks PCA CV RMSE: {np.sqrt(-xgb_walks_pca_cv_scores.mean()):.4f} ± {np.sqrt(-xgb_walks_pca_cv_scores).std():.4f}")
    
    # Train and evaluate the model
    xgb_walks_pca_model, xgb_walks_pca_preds, xgb_walks_pca_rmse = evaluate_model(
        xgb_walks_pca, X_walks_train_pca, X_walks_test_pca, y_walks_train_pca, y_walks_test_pca, "XGBoost - Walks (PCA)"
    )
    joblib.dump(xgb_walks_pca_model, 'models/xgb_walks_pca_model.joblib')
    
    # STRIKEOUTS MODELS WITH PCA
    print("\n" + "=" * 50)
    print("TRAINING STRIKEOUTS MODELS WITH PCA")
    print("=" * 50)
    
    # Random Forest for strikeouts with PCA
    rf_ks_pca = RandomForestRegressor(
        n_estimators=200,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Use cross-validation for Random Forest strikeouts model
    rf_ks_pca_cv_scores = cross_val_score(rf_ks_pca, X_ks_train_pca, y_ks_train_pca, 
                                      cv=5, scoring='neg_mean_squared_error')
    print(f"RF Strikeouts PCA CV RMSE: {np.sqrt(-rf_ks_pca_cv_scores.mean()):.4f} ± {np.sqrt(-rf_ks_pca_cv_scores).std():.4f}")
    
    # Train and evaluate the model
    rf_ks_pca_model, rf_ks_pca_preds, rf_ks_pca_rmse = evaluate_model(
        rf_ks_pca, X_ks_train_pca, X_ks_test_pca, y_ks_train_pca, y_ks_test_pca, "Random Forest - Strikeouts (PCA)"
    )
    joblib.dump(rf_ks_pca_model, 'models/rf_strikeouts_pca_model.joblib')
    
    # XGBoost for strikeouts with PCA
    xgb_ks_pca = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        gamma=0.5,
        min_child_weight=3,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Use cross-validation for XGBoost strikeouts model
    xgb_ks_pca_cv_scores = cross_val_score(xgb_ks_pca, X_ks_train_pca, y_ks_train_pca, 
                                      cv=5, scoring='neg_mean_squared_error')
    print(f"XGB Strikeouts PCA CV RMSE: {np.sqrt(-xgb_ks_pca_cv_scores.mean()):.4f} ± {np.sqrt(-xgb_ks_pca_cv_scores).std():.4f}")
    
    # Train and evaluate the model
    xgb_ks_pca_model, xgb_ks_pca_preds, xgb_ks_pca_rmse = evaluate_model(
        xgb_ks_pca, X_ks_train_pca, X_ks_test_pca, y_ks_train_pca, y_ks_test_pca, "XGBoost - Strikeouts (PCA)"
    )
    joblib.dump(xgb_ks_pca_model, 'models/xgb_strikeouts_pca_model.joblib')
    
    # Compare standard vs PCA models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON: STANDARD VS PCA")
    print("=" * 50)
    
    print("\nWalks Models:")
    print(f"RF Standard RMSE: {rf_walks_rmse:.4f} vs RF PCA RMSE: {rf_walks_pca_rmse:.4f}")
    print(f"XGB Standard RMSE: {xgb_walks_rmse:.4f} vs XGB PCA RMSE: {xgb_walks_pca_rmse:.4f}")
    
    print("\nStrikeouts Models:")
    print(f"RF Standard RMSE: {rf_ks_rmse:.4f} vs RF PCA RMSE: {rf_ks_pca_rmse:.4f}")
    print(f"XGB Standard RMSE: {xgb_ks_rmse:.4f} vs XGB PCA RMSE: {xgb_ks_pca_rmse:.4f}")
    
    # Determine best models based on RMSE
    best_walks_model = "RF Standard"
    best_walks_rmse = rf_walks_rmse
    
    if rf_walks_pca_rmse < best_walks_rmse:
        best_walks_model = "RF PCA"
        best_walks_rmse = rf_walks_pca_rmse
    
    if xgb_walks_rmse < best_walks_rmse:
        best_walks_model = "XGB Standard"
        best_walks_rmse = xgb_walks_rmse
    
    if xgb_walks_pca_rmse < best_walks_rmse:
        best_walks_model = "XGB PCA"
        best_walks_rmse = xgb_walks_pca_rmse
    
    best_ks_model = "RF Standard"
    best_ks_rmse = rf_ks_rmse
    
    if rf_ks_pca_rmse < best_ks_rmse:
        best_ks_model = "RF PCA"
        best_ks_rmse = rf_ks_pca_rmse
    
    if xgb_ks_rmse < best_ks_rmse:
        best_ks_model = "XGB Standard"
        best_ks_rmse = xgb_ks_rmse
    
    if xgb_ks_pca_rmse < best_ks_rmse:
        best_ks_model = "XGB PCA"
        best_ks_rmse = xgb_ks_pca_rmse
    
    print(f"\nBest Walks Model: {best_walks_model} (RMSE: {best_walks_rmse:.4f})")
    print(f"Best Strikeouts Model: {best_ks_model} (RMSE: {best_ks_rmse:.4f})")

# SIMULATION EXAMPLE
print("\n" + "=" * 50)
print("SIMULATION EXAMPLE")
print("=" * 50)

# Example simulation for a player
if data_loaded:
    # Use the first test sample
    player_X = X_walks_test[0:1]
    
    # If PCA models performed better, use PCA transformed data
    if pca_data_loaded and 'PCA' in best_walks_model:
        player_X = X_walks_test_pca[0:1]
    
    # Simulate walks
    walks_sims = simulate_predictions(xgb_walks_model, player_X, n_simulations=10000, noise_level=0.5)
    
    # Calculate probabilities for different lines
    for line in [0.5, 1.5, 2.5]:
        prob_over, prob_under = calculate_over_under_probs(walks_sims, line)
        print(f"Walks > {line}: {prob_over[0]:.2f}, Walks < {line}: {prob_under[0]:.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(walks_sims[0], bins=30, alpha=0.7)
    plt.title('Simulated Walks Distribution')
    plt.xlabel('Walks')
    plt.ylabel('Frequency')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Line 0.5')
    plt.axvline(x=1.5, color='g', linestyle='--', label='Line 1.5')
    plt.axvline(x=2.5, color='b', linestyle='--', label='Line 2.5')
    plt.legend()
    plt.savefig('walks_simulation.png')
    plt.close()
    
    # Example value bet calculation
    lines = [0.5, 1.5, 2.5]
    over_odds = [-110, +150, +300]
    under_odds = [-120, -180, -400]
    
    value_bets = find_value_bets(walks_sims, lines, over_odds, under_odds)
    
    if value_bets:
        print("\nValue Bets Found:")
        for bet in value_bets:
            print(f"Bet: {bet['bet']} {bet['line']} Walks, Prob: {bet['prob']:.2f}, Odds: {bet['odds']}, EV: {bet['ev']:.2f}")
    else:
        print("\nNo value bets found")
else:
    print("Skipping simulation example as we're using dummy data")

# Create a summary file with model performance
summary_path = os.path.join('models', 'model_performance_summary.txt')
with open(summary_path, 'w') as f:
    f.write("MLB PROP BETTING MODEL PERFORMANCE SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("WALKS MODELS:\n")
    f.write(f"Random Forest Standard: RMSE = {rf_walks_rmse:.4f}\n")
    f.write(f"XGBoost Standard: RMSE = {xgb_walks_rmse:.4f}\n")
    
    if pca_data_loaded:
        f.write(f"Random Forest PCA: RMSE = {rf_walks_pca_rmse:.4f}\n")
        f.write(f"XGBoost PCA: RMSE = {xgb_walks_pca_rmse:.4f}\n")
        f.write(f"Best Walks Model: {best_walks_model}\n\n")
    else:
        f.write("\n")
    
    f.write("STRIKEOUTS MODELS:\n")
    f.write(f"Random Forest Standard: RMSE = {rf_ks_rmse:.4f}\n")
    f.write(f"XGBoost Standard: RMSE = {xgb_ks_rmse:.4f}\n")
    
    if pca_data_loaded:
        f.write(f"Random Forest PCA: RMSE = {rf_ks_pca_rmse:.4f}\n")
        f.write(f"XGBoost PCA: RMSE = {xgb_ks_pca_rmse:.4f}\n")
        f.write(f"Best Strikeouts Model: {best_ks_model}\n")

print("All models trained and saved!")
print(f"Model performance summary saved to {summary_path}")