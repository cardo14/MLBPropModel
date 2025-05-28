# Add these imports if not already present
import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'src'))

# Function to load MLB data from file


def load_mlb_data():
    """Load MLB data from the MLB_Data file"""
    try:
        # Path to the MLB_Data file
        mlb_data_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'data', 'MLB_Data')

        # Load the data
        print(f"Loading MLB data from: {mlb_data_path}")
        df = pd.read_csv(mlb_data_path)

        # Process the data for our needs
        # Sort data by date to ensure proper time-based splitting
        df = df.sort_values('game_date')

        # Create a DataFrame to store processed data
        processed_data = []

        # Group by player to calculate historical stats
        player_groups = df.groupby('batting_id')

        for player_id, player_data in player_groups:
            # Sort player data by date
            player_data = player_data.sort_values('game_date')

            # For each game, use only data available before that game
            for i in range(len(player_data)):
                current_game = player_data.iloc[i:i+1]

                # Skip if this is the player's first game (no historical data)
                if i == 0:
                    continue

                # Get historical data for this player (all games before current)
                historical_data = player_data.iloc[:i]

                # Calculate historical stats (only using data from previous games)
                # Basic batting stats
                hist_batting_avg = historical_data['batting_hits'].sum(
                ) / max(1, historical_data['batting_atBats'].sum())
                hist_obp = historical_data['batting_baseOnBalls'].sum(
                ) / max(1, historical_data['batting_plateAppearances'].sum())

                # Walks-specific metrics
                hist_walks = historical_data['batting_walks'].sum(
                ) / max(1, len(historical_data))
                hist_walk_rate = historical_data['batting_baseOnBalls'].sum(
                ) / max(1, historical_data['batting_plateAppearances'].sum())
                hist_walks_last_10 = historical_data.iloc[-10:]['batting_walks'].sum() / min(
                    10, len(historical_data))

                # Strikeouts-specific metrics
                hist_strikeouts = historical_data['batting_strikeouts'].sum(
                ) / max(1, len(historical_data))
                hist_k_rate = historical_data['batting_strikeOuts'].sum(
                ) / max(1, historical_data['batting_plateAppearances'].sum())
                hist_ks_last_10 = historical_data.iloc[-10:]['batting_strikeouts'].sum() / min(
                    10, len(historical_data))

                # Batting order metrics
                bo = current_game['batting_battingOrder']
                bo = bo // 100
                is_pinch_hitter = bo % 100 == 1

                # Get pitcher data for current game
                pitcher_id = current_game['pitching_id'].values[0]

                # Get historical data for this pitcher
                pitcher_data = df[df['pitching_id'] == pitcher_id]
                pitcher_data = pitcher_data[pitcher_data['game_date']
                                            < current_game['game_date'].values[0]]

                # Calculate pitcher historical stats
                if len(pitcher_data) > 0:
                    # Basic pitcher stats
                    pitcher_hist_era = pitcher_data['pitching_era'].mean()
                    pitcher_hist_whip = pitcher_data['pitching_whip'].mean()

                    # Strikeout-specific pitcher metrics
                    pitcher_hist_k_per_9 = 9 * pitcher_data['pitching_strikeOuts'].sum() / max(
                        1, pitcher_data['pitching_inningsPitched'].sum())
                    pitcher_hist_k_rate = pitcher_data['pitching_strikeOuts'].sum(
                    ) / max(1, pitcher_data['pitching_battersFaced'].sum())
                    pitcher_recent_k_per_9 = 9 * pitcher_data.iloc[-3:]['pitching_strikeOuts'].sum(
                    ) / max(1, pitcher_data.iloc[-3:]['pitching_inningsPitched'].sum())

                    # Walk-specific pitcher metrics
                    pitcher_hist_bb_per_9 = 9 * pitcher_data['pitching_baseOnBalls'].sum() / max(
                        1, pitcher_data['pitching_inningsPitched'].sum())
                    pitcher_hist_bb_rate = pitcher_data['pitching_baseOnBalls'].sum(
                    ) / max(1, pitcher_data['pitching_battersFaced'].sum())
                    pitcher_recent_bb_per_9 = 9 * pitcher_data.iloc[-3:]['pitching_baseOnBalls'].sum(
                    ) / max(1, pitcher_data.iloc[-3:]['pitching_inningsPitched'].sum())

                    # Control metrics
                    pitcher_hist_k_bb_ratio = pitcher_data['pitching_strikeOuts'].sum(
                    ) / max(1, pitcher_data['pitching_baseOnBalls'].sum())
                else:
                    # Use league averages if no historical data
                    pitcher_hist_era = 4.0
                    pitcher_hist_whip = 1.3
                    pitcher_hist_k_per_9 = 8.0
                    pitcher_hist_bb_per_9 = 3.0
                    pitcher_hist_k_rate = 0.22
                    pitcher_hist_bb_rate = 0.08
                    pitcher_recent_k_per_9 = 8.0
                    pitcher_recent_bb_per_9 = 3.0
                    pitcher_hist_k_bb_ratio = 2.7

                # Create row with current game info and historical stats
                game_row = {
                    'player_id': player_id,
                    'game_id': current_game['game_id'].values[0],
                    'date': pd.to_datetime(current_game['game_date'].values[0]),
                    'venue': current_game['game_venue'].values[0],
                    'is_home_game': int(current_game['batting_battingOrder'].values[0]) < 500,

                    # TARGET VARIABLES - what we're predicting
                    'walks': current_game['batting_walk_recorded'].values[0],
                    'strikeouts': current_game['batting_strikeout_recorded'].values[0],

                    # HISTORICAL PLAYER STATS - available before the game
                    'hist_batting_avg': hist_batting_avg,
                    'hist_obp': hist_obp,
                    'games_played': i,  # Number of previous games

                    # WALKS-SPECIFIC PLAYER STATS
                    'hist_walks_per_game': hist_walks,
                    'hist_walk_rate': hist_walk_rate,
                    'hist_walks_last_10': hist_walks_last_10,

                    # BATTING ORDER STATS
                    'batting_position': bo,
                    'pinch_status': is_pinch_hitter,

                    # STRIKEOUTS-SPECIFIC PLAYER STATS
                    'hist_strikeouts_per_game': hist_strikeouts,
                    'hist_k_rate': hist_k_rate,
                    'hist_ks_last_10': hist_ks_last_10,

                    # HISTORICAL PITCHER STATS - available before the game
                    'pitcher_id': pitcher_id,
                    'pitcher_hist_era': pitcher_hist_era,
                    'pitcher_hist_whip': pitcher_hist_whip,

                    # STRIKEOUT-SPECIFIC PITCHER STATS
                    'pitcher_hist_k_per_9': pitcher_hist_k_per_9,
                    'pitcher_hist_k_rate': pitcher_hist_k_rate,
                    'pitcher_recent_k_per_9': pitcher_recent_k_per_9,

                    # WALK-SPECIFIC PITCHER STATS
                    'pitcher_hist_bb_per_9': pitcher_hist_bb_per_9,
                    'pitcher_hist_bb_rate': pitcher_hist_bb_rate,
                    'pitcher_recent_bb_per_9': pitcher_recent_bb_per_9,
                    'pitcher_hist_k_bb_ratio': pitcher_hist_k_bb_ratio,

                    # GAME CONTEXT FEATURES
                    'venue_is_dome': 'Dome' in current_game['game_venue'].values[0] or 'Field' in current_game['game_venue'].values[0],
                    'temperature': current_game['temp'].values[0] if 'temp' in current_game.columns else 70.0
                }

                processed_data.append(game_row)

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)

        # Drop rows with NaN values
        processed_df = processed_df.dropna()

        return processed_df

    except Exception as e:
        print(f"Error loading MLB data: {e}")
        return None

# Enhanced preprocessing function with optional PCA


def enhanced_preprocess_data(df, target_col, feature_engineering=True, apply_pca=False, n_components=0.95):
    """Enhanced preprocessing with feature engineering and time-based splitting"""
    # Handle missing values
    df = df.fillna(0)

    # Sort by date to ensure proper time-based splitting
    df = df.sort_values('date')

    # Define features based on the processed data structure
    if target_col == 'walks':
        features = [
            # Basic player stats
            'hist_batting_avg', 'hist_obp', 'games_played',

            # Walk-specific player stats
            'hist_walks_per_game', 'hist_walk_rate', 'hist_walks_last_10',

            # Batting order stats
            'batting_position', 'pinch_status',

            # Game context
            'is_home_game', 'venue_is_dome', 'temperature',

            # Basic pitcher stats
            'pitcher_hist_era', 'pitcher_hist_whip',

            # Walk-specific pitcher stats
            'pitcher_hist_bb_per_9', 'pitcher_hist_bb_rate', 'pitcher_recent_bb_per_9',
            'pitcher_hist_k_bb_ratio'
        ]
    else:  # strikeouts
        features = [
            # Basic player stats
            'hist_batting_avg', 'hist_obp', 'games_played',

            # Strikeout-specific player stats
            'hist_strikeouts_per_game', 'hist_k_rate', 'hist_ks_last_10',

            # Game context
            'is_home_game', 'venue_is_dome', 'temperature',

            # Batting order stats
            'batting_position', 'pinch_status',

            # Basic pitcher stats
            'pitcher_hist_era', 'pitcher_hist_whip',

            # Strikeout-specific pitcher stats
            'pitcher_hist_k_per_9', 'pitcher_hist_k_rate', 'pitcher_recent_k_per_9',
            'pitcher_hist_k_bb_ratio'
        ]

    # Add categorical features if available
    categorical_features = []
    for col in ['venue', 'batting_position']:
        if col in df.columns:
            categorical_features.append(col)

    # Combine features
    all_features = features + categorical_features

    # Check if all features exist in the dataframe
    existing_features = [f for f in all_features if f in df.columns]

    # Create X and y
    X = df[existing_features]
    y = df[target_col]

    # Create time-based train/test split (75/25)
    # Use the last 25% of data (chronologically) for testing
    split_idx = int(len(df) * 0.75)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")

    # Create preprocessing pipeline
    numeric_features = [
        f for f in existing_features if f not in categorical_features]

    # Define transformers
    numeric_transformer = StandardScaler()

    # Create preprocessor
    if categorical_features and len(categorical_features) > 0:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

    # Create pipeline steps
    pipeline_steps = [('preprocessor', preprocessor)]

    # Add PCA if requested
    if apply_pca:
        # Add PCA step to the pipeline
        pipeline_steps.append(('pca', PCA(n_components=n_components)))

    # Create and fit the pipeline
    pipeline = Pipeline(steps=pipeline_steps)
    pipeline.fit(X_train)

    # Transform the data
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # If PCA was applied, analyze and print component information
    if apply_pca and hasattr(pipeline, 'named_steps') and 'pca' in pipeline.named_steps:
        pca = pipeline.named_steps['pca']
        n_components = pca.n_components_
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        print(
            f"\nPCA applied with {n_components} components, explaining {cumulative_variance[-1]:.4f} of variance")
        print(
            f"Explained variance ratio per component: {explained_variance[:5]}...")

        # Analyze what features contribute to principal components
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
            analyze_principal_components(pca, feature_names, target_col)

    # Save the pipeline for later use
    model_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save standard pipeline
    pipeline_path = os.path.join(model_dir, f'{target_col}_pipeline.joblib')
    joblib.dump(pipeline, pipeline_path)

    return X, y, X_train_processed, X_test_processed, y_train, y_test, pipeline


# Get data for modeling
print("Loading MLB data for modeling...")

# Load data from MLB_Data file
mlb_data = load_mlb_data()

# If data loading fails, create sample data as fallback
if mlb_data is None or mlb_data.empty:
    print("Error loading MLB data. Creating sample data for demonstration.")
    # Create sample data with realistic MLB stats
    sample_data = pd.DataFrame({
        'player_id': range(100),
        'first_name': ['Player'] * 100,
        'last_name': [f'Sample_{i}' for i in range(100)],
        'position': ['B'] * 100,
        'date': pd.to_datetime('2025-05-01'),
        'walks': np.random.randint(0, 5, 100),
        'strikeouts': np.random.randint(0, 10, 100),
        'at_bats': np.random.randint(3, 6, 100),
        'hits': np.random.randint(0, 4, 100),
        'batting_avg': np.random.uniform(0.2, 0.35, 100),
        'on_base_pct': np.random.uniform(0.3, 0.45, 100),
        'slugging_pct': np.random.uniform(0.35, 0.55, 100),
        'ops': np.random.uniform(0.65, 1.0, 100),
        'game_id': range(100),
        'venue': ['Sample Stadium'] * 100,
        'venue_is_dome': [False] * 100,
        'home_team_id': [1] * 100,
        'away_team_id': [2] * 100,
        'is_home_game': [1] * 50 + [0] * 50,
        'pitcher_id': np.random.randint(1000, 1100, 100),
        'pitcher_strikeouts': np.random.randint(100, 250, 100),
        'pitcher_walks': np.random.randint(30, 100, 100),
        'era': np.random.uniform(3.0, 5.0, 100),
        'whip': np.random.uniform(1.0, 1.5, 100)
    })
    mlb_data = sample_data

print(f"Loaded {len(mlb_data)} MLB data records")

# Use the same data for both walks and strikeouts models
walks_data = mlb_data.copy()
strikeouts_data = mlb_data.copy()

# Function to analyze principal components


def analyze_principal_components(pca, feature_names, target_col):
    """Analyze what each principal component represents"""
    # Get component loadings
    loadings = pca.components_

    # Print importance of each feature to each principal component
    print(f"\nPrincipal Component Analysis for {target_col.capitalize()}:")
    for i, component in enumerate(loadings[:5]):  # First 5 components or fewer
        if i >= len(pca.explained_variance_ratio_):
            break

        print(
            f"\nPrincipal Component {i+1} (Explains {pca.explained_variance_ratio_[i]:.4f} of variance)")

        # Sort features by absolute importance
        sorted_idx = np.argsort(np.abs(component))[::-1]

        # Print top 5 features for this component
        for j in sorted_idx[:5]:
            if j < len(feature_names):
                print(f"  {feature_names[j]}: {component[j]:.4f}")


# Preprocess walks data with enhanced preprocessing (standard version)
X_walks, y_walks, X_walks_train, X_walks_test, y_walks_train, y_walks_test, walks_pipeline = \
    enhanced_preprocess_data(walks_data, 'walks', apply_pca=False)

# Preprocess strikeouts data with enhanced preprocessing (standard version)
X_ks, y_ks, X_ks_train, X_ks_test, y_ks_train, y_ks_test, ks_pipeline = \
    enhanced_preprocess_data(strikeouts_data, 'strikeouts', apply_pca=False)

# Also create PCA versions for comparison
print("\nCreating PCA versions of the preprocessed data...")

# For PCA versions, we still use the original target columns ('walks' and 'strikeouts')
# but we'll save the pipelines with different names
X_walks_pca, y_walks_pca, X_walks_train_pca, X_walks_test_pca, y_walks_train_pca, y_walks_test_pca, walks_pca_pipeline = \
    enhanced_preprocess_data(
        walks_data, 'walks', apply_pca=True, n_components=0.95)

# Save the PCA pipeline with a different name
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(walks_pca_pipeline, os.path.join(
    model_dir, 'walks_pca_pipeline.joblib'))

X_ks_pca, y_ks_pca, X_ks_train_pca, X_ks_test_pca, y_ks_train_pca, y_ks_test_pca, ks_pca_pipeline = \
    enhanced_preprocess_data(
        strikeouts_data, 'strikeouts', apply_pca=True, n_components=0.95)

# Save the PCA pipeline with a different name
joblib.dump(ks_pca_pipeline, os.path.join(
    model_dir, 'strikeouts_pca_pipeline.joblib'))
