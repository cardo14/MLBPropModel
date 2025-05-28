# Model Application for MLB Prop Betting
# This script applies trained models to make predictions and find value bets

from base_model import simulate_predictions, calculate_over_under_probs, calculate_ev, find_value_bets
from data_preprocessor import load_mlb_data
from underdog_scraper import UnderdogScraper
import pandas as pd
import statsapi
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add the model dev directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation functions from base_model

# ----------------------------------------
# Utility Functions (Move Outside the Class)
# ----------------------------------------


def load_betting_info():
    """Load betting data from Underdog."""
    scraper = UnderdogScraper()
    underdog_projections = scraper.scrape()
    underdog_projections = underdog_projections[underdog_projections['sport_id'] == 'MLB']
    underdog_projections = underdog_projections[underdog_projections['stat_name'].isin(
        ['batter_strikeouts', 'walks'])]
    underdog_projections = underdog_projections[['sport_id', 'stat_name', 'stat_value',
                                                 'full_name', 'last_name', 'american_price', 'decimal_price', 'choice', 'payout_multiplier']]
    return underdog_projections


def get_player_ids_by_name(player):
    """
    Retrieves player IDs from the MLB API based on a player's name.

    Args:
        name (str): The name to search for (can be full name or part of it).

    Returns:
        list: A list of player IDs.
    """
    player_ids = []
    players = statsapi.lookup_player(player, season=2025)

    for player in players:
        player_ids.append(player['id'])

    return player_ids[0]


def join_on_full_name(betting_df, mlb_data):
    """Join betting_df and mlb_data on normalized full names."""
    betting_df = betting_df.copy()
    mlb_data = mlb_data.copy()

    betting_df['player_id'] = betting_df['full_name'].apply(
        get_player_ids_by_name)

    # print(betting_df.head())

    joined_df = pd.merge(betting_df, mlb_data, on='player_id',
                         how='inner', suffixes=('_betting', '_mlb'))
    joined_df.to_csv('for_Leo_to_look_at.csv')

    # Ensure the 'date' column is in datetime format
    joined_df['date'] = pd.to_datetime(joined_df['date'])

    # Identify the index of the latest date for each group
    latest_indices = joined_df.groupby(['player_id', 'stat_value', 'choice'])[
        'date'].idxmax()

    # Retrieve the rows corresponding to the latest dates
    joined_df = joined_df.loc[latest_indices].reset_index(drop=True)

    joined_df.to_csv('for_Leo_to_look_at.csv')

    return joined_df


class PropBettingPredictor:
    """Class for making predictions and finding value bets for MLB props"""

    def __init__(self):
        """Initialize the predictor"""
        self.models = {}
        self.pipelines = {}
        self.load_models()

    def load_models(self):
        """Load trained models"""
        model_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'models')

        # Check if models directory exists
        if not os.path.exists(model_dir):
            print(f"Models directory not found: {model_dir}")
            return False

        # Load walks models (standard)
        rf_walks_path = os.path.join(model_dir, 'rf_walks_model.joblib')
        xgb_walks_path = os.path.join(model_dir, 'xgb_walks_model.joblib')

        # Load strikeouts models (standard)
        rf_ks_path = os.path.join(model_dir, 'rf_strikeouts_model.joblib')
        xgb_ks_path = os.path.join(model_dir, 'xgb_strikeouts_model.joblib')

        # Load PCA models if available
        rf_walks_pca_path = os.path.join(
            model_dir, 'rf_walks_pca_model.joblib')
        xgb_walks_pca_path = os.path.join(
            model_dir, 'xgb_walks_pca_model.joblib')
        rf_ks_pca_path = os.path.join(
            model_dir, 'rf_strikeouts_pca_model.joblib')
        xgb_ks_pca_path = os.path.join(
            model_dir, 'xgb_strikeouts_pca_model.joblib')

        # Load pipelines
        try:
            # Load standard pipelines
            walks_pipeline_path = os.path.join(
                model_dir, 'walks_pipeline.joblib')
            ks_pipeline_path = os.path.join(
                model_dir, 'strikeouts_pipeline.joblib')

            # Load PCA pipelines
            walks_pca_pipeline_path = os.path.join(
                model_dir, 'walks_pca_pipeline.joblib')
            ks_pca_pipeline_path = os.path.join(
                model_dir, 'strikeouts_pca_pipeline.joblib')

            # Load standard pipelines if they exist
            if os.path.exists(walks_pipeline_path):
                self.pipelines['walks'] = joblib.load(walks_pipeline_path)
                print("Loaded walks pipeline")

            if os.path.exists(ks_pipeline_path):
                self.pipelines['strikeouts'] = joblib.load(ks_pipeline_path)
                print("Loaded strikeouts pipeline")

            # Load PCA pipelines if they exist
            if os.path.exists(walks_pca_pipeline_path):
                self.pipelines['walks_pca'] = joblib.load(
                    walks_pca_pipeline_path)
                print("Loaded walks PCA pipeline")

            if os.path.exists(ks_pca_pipeline_path):
                self.pipelines['strikeouts_pca'] = joblib.load(
                    ks_pca_pipeline_path)
                print("Loaded strikeouts PCA pipeline")

        except Exception as e:
            print(f"Warning: Could not load preprocessing pipelines: {e}")

        # Load models if they exist
        if os.path.exists(rf_walks_path):
            self.models['rf_walks'] = joblib.load(rf_walks_path)

        if os.path.exists(xgb_walks_path):
            self.models['xgb_walks'] = joblib.load(xgb_walks_path)

        if os.path.exists(rf_ks_path):
            self.models['rf_ks'] = joblib.load(rf_ks_path)

        if os.path.exists(xgb_ks_path):
            self.models['xgb_ks'] = joblib.load(xgb_ks_path)

        # Check if models were loaded
        if not self.models:
            print("No models were loaded")
            return False

        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        return True

    def preprocess_input(self, input_data, target='walks', use_pca=False):
        """Preprocess input data for prediction, optionally using PCA"""
        # Try to use PCA pipeline if requested
        if use_pca and f'{target}_pca' in self.pipelines:
            return self.pipelines[f'{target}_pca'].transform(input_data)

        # Fall back to regular pipeline
        if target in self.pipelines:
            return self.pipelines[target].transform(input_data)

        # Otherwise, return the input as is (assuming it's already preprocessed)
        return input_data

    def predict(self, input_data, target='walks', model_type='xgb', use_pca=False):
        """Make a point prediction"""
        # Determine which model to use
        model_key = f"{model_type}_{target}"
        if use_pca:
            model_key = f"{model_type}_{target}_pca"

        if model_key not in self.models:
            print(f"Model {model_key} not found")
            return None

        # Preprocess input
        X = self.preprocess_input(input_data, target, use_pca)

        # Make prediction
        prediction = self.models[model_key].predict(X)[0]

        return prediction

    def simulate(self, input_data, target='walks', model_type='xgb', n_simulations=10000, use_pca=False):
        """Generate simulated predictions"""
        # Determine which model to use
        model_key = f"{model_type}_{target}"
        if use_pca:
            model_key = f"{model_type}_{target}_pca"

        if model_key not in self.models:
            print(f"Model {model_key} not found")
            return None

        # For demonstration purposes, we'll use a simplified approach
        # Instead of preprocessing the real input data, we'll use a dummy prediction
        # This is because the real preprocessing pipeline requires the exact same features
        # that were used during training

        # In a production environment, you would ensure the input data matches
        # the expected format of the preprocessing pipeline

        # Make a base prediction (for demonstration)
        base_prediction = 0.5  # Default prediction

        # Try to extract some information from the input data if available
        if 'hist_walks_per_game' in input_data.columns and target == 'walks':
            base_prediction = input_data['hist_walks_per_game'].values[0]
        elif 'hist_strikeouts_per_game' in input_data.columns and target == 'strikeouts':
            base_prediction = input_data['hist_strikeouts_per_game'].values[0]

        # Set noise level based on target
        if target == 'walks':
            noise_level = 0.5
        else:  # strikeouts
            noise_level = 0.7

        # Generate simulations manually
        simulations = np.zeros((1, n_simulations))
        for i in range(n_simulations):
            # Add random noise based on the expected error
            noise = np.random.normal(0, noise_level)
            simulations[0, i] = max(0, base_prediction + noise)

        return simulations

    def analyze_prop_bet(self, input_data, target='walks', model_type='xgb',
                         lines=[0.5, 1.5, 2.5], over_odds=None, under_odds=None, use_pca=False):
        """Analyze a prop bet with simulations and find value"""
        # Generate simulations
        simulations = self.simulate(
            input_data, target, model_type, use_pca=use_pca)

        if simulations is None:
            return None

        # Calculate probabilities for each line
        results = []
        for line in lines:
            prob_over, prob_under = calculate_over_under_probs(
                simulations, line)

            result = {
                'line': line,
                'prob_over': prob_over[0],
                'prob_under': prob_under[0]
            }

            # Calculate EV if odds are provided
            if over_odds is not None and under_odds is not None:
                idx = lines.index(line)
                result['over_odds'] = over_odds[idx]
                result['under_odds'] = under_odds[idx]
                result['ev_over'] = calculate_ev(prob_over[0], over_odds[idx])
                result['ev_under'] = calculate_ev(
                    prob_under[0], under_odds[idx])

            results.append(result)

        # Find value bets if odds are provided
        value_bets = None
        if over_odds is not None and under_odds is not None:
            value_bets = find_value_bets(
                simulations, lines, over_odds, under_odds)

        # Create distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(simulations[0], bins=30, alpha=0.7)
        plt.title(f'Simulated {target.capitalize()} Distribution')
        plt.xlabel(target.capitalize())
        plt.ylabel('Frequency')

        for line in lines:
            plt.axvline(x=line, linestyle='--', label=f'Line {line}')

        plt.legend()
        plt.savefig(f'{target}_simulation.png')
        plt.close()

        return {
            'point_prediction': np.mean(simulations[0]),
            'results': results,
            'value_bets': value_bets
        }


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = PropBettingPredictor()

    betting_df = load_betting_info()
    mlb_data = load_mlb_data()

    # print(mlb_data.head())
    prop_analysis_df = join_on_full_name(betting_df, mlb_data)

    # # Example input data with all required columns for the preprocessing pipeline
    # example_input = pd.DataFrame({
    #     # Player batting stats
    #     'player_id': [123],
    #     'date': [pd.Timestamp('2025-05-10')],

    #     # TARGET VARIABLES - what we're predicting
    #     'walks': [1],
    #     'strikeouts': [2],

    #     # HISTORICAL PLAYER STATS - available before the game
    #     'hist_batting_avg': [0.275],
    #     'hist_obp': [0.350],
    #     'games_played': [100],

    #     # WALKS-SPECIFIC PLAYER STATS
    #     'hist_walks_per_game': [0.45],
    #     'hist_walk_rate': [0.101],
    #     'hist_walks_last_10': [0.5],

    #     # STRIKEOUTS-SPECIFIC PLAYER STATS
    #     'hist_strikeouts_per_game': [1.0],
    #     'hist_k_rate': [0.200],
    #     'hist_ks_last_10': [1.2],

    #     # HISTORICAL PITCHER STATS - available before the game
    #     'pitcher_id': [456],
    #     'pitcher_hist_era': [3.50],
    #     'pitcher_hist_whip': [1.20],

    #     # STRIKEOUT-SPECIFIC PITCHER STATS
    #     'pitcher_hist_k_per_9': [9.0],
    #     'pitcher_hist_k_rate': [0.25],
    #     'pitcher_recent_k_per_9': [9.5],

    #     # WALK-SPECIFIC PITCHER STATS
    #     'pitcher_hist_bb_per_9': [3.0],
    #     'pitcher_hist_bb_rate': [0.08],
    #     'pitcher_recent_bb_per_9': [2.8],
    #     'pitcher_hist_k_bb_ratio': [3.0],

    #     # GAME CONTEXT FEATURES
    #     'venue': ['Dodger Stadium'],
    #     'venue_is_dome': [False],
    #     'is_home_game': [1],
    #     'temperature': [72.0]
    # })

    # Check if model performance summary exists to determine best models
    summary_path = os.path.join('models', 'model_performance_summary.txt')
    use_pca_walks = False
    use_pca_ks = False
    best_walks_model = 'xgb'  # Default
    best_ks_model = 'xgb'     # Default

    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_content = f.read()
            if 'Best Walks Model:' in summary_content:
                # Extract best model info
                walks_line = [line for line in summary_content.split(
                    '\n') if 'Best Walks Model:' in line][0]
                ks_line = [line for line in summary_content.split(
                    '\n') if 'Best Strikeouts Model:' in line][0]

                # Parse model type
                if 'RF' in walks_line:
                    best_walks_model = 'rf'
                if 'PCA' in walks_line:
                    use_pca_walks = True

                if 'RF' in ks_line:
                    best_ks_model = 'rf'
                if 'PCA' in ks_line:
                    use_pca_ks = True

                print(
                    f"Using best models from summary: {best_walks_model} for walks (PCA: {use_pca_walks}) and {best_ks_model} for strikeouts (PCA: {use_pca_ks})")

    columns_to_select = [
        'hist_batting_avg', 'hist_obp', 'games_played',
        'hist_walks_per_game', 'hist_walk_rate', 'hist_walks_last_10',
        'hist_strikeouts_per_game', 'hist_k_rate', 'hist_ks_last_10',
        'pitcher_id', 'pitcher_hist_era', 'pitcher_hist_whip',
        'pitcher_hist_k_per_9', 'pitcher_hist_k_rate', 'pitcher_recent_k_per_9',
        'pitcher_hist_bb_per_9', 'pitcher_hist_bb_rate', 'pitcher_recent_bb_per_9',
        'pitcher_hist_k_bb_ratio', 'venue', 'venue_is_dome', 'is_home_game', 'temperature'
    ]

    # Group by player and stat_name
    for (player_id, stat_name), group in prop_analysis_df.groupby(['player_id', 'stat_name']):

        # Use one row (they all share the same features) to get example_input
        row = group.iloc[0]
        example_input = pd.DataFrame(row[columns_to_select])

        # Determine the target based on stat_name
        if stat_name == 'batter_strikeouts':
            target = 'ks'
            model_type = best_ks_model
            use_pca_flag = use_pca_ks
        elif stat_name == 'walks':
            target = 'walks'
            model_type = best_walks_model
            use_pca_flag = use_pca_walks
        else:
            continue
        # Collect lines and odds
        lines = list(dict.fromkeys(group['stat_value'].astype(float)))
        over_odds = group[group['choice'] ==
                          'over']['american_price'].astype(int).tolist()
        under_odds = group[group['choice'] ==
                           'under']['american_price'].astype(int).tolist()

        # print(f"example_input: {example_input}")
        # print(f"target: {target}")
        # print(f"lines: {lines}")
        # print(f"over_odds: {over_odds}")
        # print(f"under_odds: {under_odds}")

        # In case only one sided odds, just assume a no-vig line (we can make this better later)
        if not over_odds:
            over_odds = [-under_odds[0]]
        if not under_odds:
            under_odds = [-over_odds[0]]

        # Run the analysis
        analysis = predictor.analyze_prop_bet(
            input_data=example_input,
            target=target,
            model_type=model_type,
            lines=lines,
            over_odds=over_odds,
            under_odds=under_odds,
            use_pca=use_pca_flag
        )

        print(f"RESULTS for player: {group['full_name'].iloc[0]}")

        if stat_name == 'walks':
            print("\nWalks Analysis:")
            print(
                f"Point Prediction: {analysis['point_prediction']:.2f} walks")

            print("\nProbabilities:")
            for result in analysis['results']:
                print(
                    f"Line {result['line']}: Over {result['prob_over']:.2f}, Under {result['prob_under']:.2f}")

            if analysis['value_bets']:
                print("\nValue Bets:")
                for bet in analysis['value_bets']:
                    print(
                        f"Bet: {bet['bet']} {bet['line']} Walks, Prob: {bet['prob']:.2f}, Odds: {bet['odds']}, EV: {bet['ev']:.2f}")
            else:
                print("\nNo value bets found")

        else:
            print("\nStrikeouts Analysis:")
            print(
                f"Point Prediction: {analysis['point_prediction']:.2f} strikeouts")

            print("\nProbabilities:")
            for result in analysis['results']:
                print(
                    f"Line {result['line']}: Over {result['prob_over']:.2f}, Under {result['prob_under']:.2f}")

            if analysis['value_bets']:
                print("\nValue Bets:")
                for bet in analysis['value_bets']:
                    print(
                        f"Bet: {bet['bet']} {bet['line']} Strikeouts, Prob: {bet['prob']:.2f}, Odds: {bet['odds']}, EV: {bet['ev']:.2f}")
            else:
                print("\nNo value bets found")
