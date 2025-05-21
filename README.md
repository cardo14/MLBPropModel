# MLB Prop Betting System

A comprehensive system for analyzing MLB player statistics and generating prop betting predictions using machine learning models with Principal Component Analysis (PCA) integration.

## Project Structure

The project has been organized into a logical structure:

```
├── data/                  # Data files and CSV exports
│   └── MLB_Data/          # MLB player and game statistics
├── legacy_scripts/        # Original data collection scripts
├── model_dev/             # Model development and application
│   ├── base_model.py      # Core model training and evaluation
│   ├── data_preprocessor.py # Data preprocessing with PCA integration
│   ├── model_application.py # Model application for predictions
│   └── models/            # Saved models and pipelines
├── scripts/               # Utility scripts for data import and maintenance
├── src/                   # Source code
│   ├── api/               # API endpoints and interfaces
│   ├── data/              # Data processing and management
│   ├── models/            # Prediction models and betting analysis
│   └── utils/             # Utility functions and configuration
└── config.ini            # Configuration file
```

### Key Files

#### Model Development
- `model_dev/data_preprocessor.py` - Data preprocessing with feature engineering and PCA integration
- `model_dev/base_model.py` - Machine learning model training, evaluation, and comparison
- `model_dev/model_application.py` - Model application for predictions and prop bet analysis

#### Legacy System
- `main.py` - Main application entry point with menu-driven interface
- `src/utils/config.py` - Configuration management and logging setup
- `src/utils/db_utils.py` - Database connection and schema management
- `src/models/betting.py` - Functions for prop betting odds, predictions, and value bet analysis
- `scripts/import_dataframe.py` - Script to import CSV data into the database

## Setup

1. Create a PostgreSQL database named `mlb_prop_betting`
2. Create a `config.ini` file with the following structure:

```ini
[Database]
host = localhost
database = mlb_prop_betting
user = postgres
password = your_password

[API]
base_url = https://statsapi.mlb.com/api/v1
sportsbook_api_key = your_sportsbook_api_key
sportsbook_api_url = https://sportsbook-api-url.com
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Data Import

To import your CSV data into the database:

```bash
python scripts/import_dataframe.py --csv path/to/your/dataframe.csv --reset
```

Options:
- `--csv`: Path to your CSV dataframe (defaults to `data/example_data.csv`)
- `--reset`: Optional flag to reset the database tables before importing

## Usage

### Running the Machine Learning Models

To train and evaluate the models with PCA integration:

```bash
cd model_dev
python base_model.py
```

This will:
1. Load and preprocess the MLB data
2. Apply PCA to create dimensionality-reduced versions of the data
3. Train and evaluate Random Forest and XGBoost models for walks and strikeouts
4. Compare standard vs. PCA models and select the best performers
5. Save all models and pipelines to the models directory

To make predictions using the trained models:

```bash
cd model_dev
python model_application.py
```

This will:
1. Load the best models from the previous training
2. Generate predictions for sample player data
3. Analyze the predictions for prop betting opportunities
4. Calculate probabilities and expected values for different betting lines

### Running the Legacy Application

Run the main application:

```bash
python main.py
```

The application provides a menu-driven interface with the following options:

1. **Data Management**
   - View player statistics
   - View game schedule
   - View prop betting odds
   - Import data from CSV

2. **Analysis & Predictions**
   - Generate predictions
   - Find value bets
   - Export predictions to CSV

3. **Database Management**
   - View database status
   - Reset database tables

## Database Schema

The database is structured with the following tables:

- `teams`: MLB team information
- `players`: Player details and team affiliations
- `player_batting_stats`: Batting statistics for players
- `player_pitching_stats`: Pitching statistics for players
- `games`: Game schedules and venue information
- `prop_types`: Types of prop bets (hits, home runs, strikeouts, etc.)
- `sportsbooks`: Sportsbook information
- `odds`: Prop betting odds from sportsbooks
- `predictions`: Model predictions for player props

## Machine Learning Models and PCA Integration

### Models

The system uses two types of machine learning models for predicting player props:

1. **Random Forest**: Ensemble learning method that builds multiple decision trees and merges them for more accurate predictions.
2. **XGBoost**: Gradient boosting algorithm that provides high performance and efficiency for regression tasks.

Both models are trained separately for:
- **Walks Prediction**: Predicting the number of walks a batter will take in a game
- **Strikeouts Prediction**: Predicting the number of strikeouts a batter will have in a game

### Principal Component Analysis (PCA)

PCA has been integrated into the preprocessing pipeline to:

1. **Reduce Dimensionality**: Transform the original features into a smaller set of uncorrelated principal components
2. **Improve Model Performance**: Potentially improve prediction accuracy by removing noise and multicollinearity
3. **Feature Insights**: Provide insights into which features drive the most variance in the data

The system trains and evaluates both standard models and PCA-based models, then selects the best performing model for each prediction target.

### Feature Engineering

The system uses a rich set of features including:

- **Player Historical Stats**: Batting average, OBP, walk rate, strikeout rate
- **Recent Performance**: Last 10 games for batters, last 3 starts for pitchers
- **Pitcher Metrics**: ERA, WHIP, K/9, BB/9, K/BB ratio
- **Game Context**: Home/away, temperature, venue type (dome/outdoor)

### Model Evaluation

Models are evaluated using multiple metrics:

- **RMSE**: Root Mean Squared Error for overall prediction accuracy
- **MAE**: Mean Absolute Error for average prediction error
- **R²**: Coefficient of determination for explained variance
- **Accuracy within 0.5 units**: Percentage of predictions within 0.5 units of actual value
- **Over/Under Accuracy**: Accuracy of predicting whether a player will go over or under a specific line

## Data Flow

1. **Data Preprocessing**: Raw MLB data is processed with feature engineering and optional PCA
2. **Model Training**: Multiple models (RF, XGBoost) are trained with and without PCA
3. **Model Comparison**: Models are compared to select the best performer for each target
4. **Prediction Generation**: Best models generate predictions for player props
5. **Value Identification**: Predictions are compared with betting odds to find value bets

## Notes

### Key Findings from PCA Integration

- **Walks Prediction**: Pitcher walk tendencies (BB/9, BB rate) and batter walk history are the most important factors
- **Strikeouts Prediction**: Pitcher strikeout ability (K/9, K rate) and batter strikeout history drive predictions
- **Model Performance**: Standard XGBoost model performed best for walks prediction (RMSE: 0.4280), while standard Random Forest performed best for strikeouts prediction (RMSE: 0.4961)
- **PCA Insights**: While PCA models didn't significantly improve RMSE, they provided valuable insights into feature importance and slightly improved over/under prediction accuracy for strikeouts

### Future Enhancements

- **Additional Features**: Incorporate matchup-specific statistics (batter vs. pitcher history)
- **Advanced Models**: Implement neural networks or ensemble methods combining standard and PCA models
- **Real-time Predictions**: Develop a pipeline for daily predictions based on upcoming games
- **Odds Integration**: Automate the collection of current betting odds for comparison with model predictions

### Legacy System

- The original data collection scripts are preserved in the `legacy_scripts` directory
- The prediction model can be enhanced with more sophisticated statistical methods
- Always use caution when using the "Reset database tables" option as it will delete all existing data
