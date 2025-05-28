# Database Update Scripts

This directory contains all scripts responsible for updating the MLB statistics database with fresh data.

## Scripts Overview

### `daily_database_update.py` (Main Script)
- **Purpose**: Primary script that orchestrates the daily database update process
- **Usage**: Called by cron job nightly at 1:00 AM
- **Function**: Updates all MLB data including games, teams, players, and statistics
- **Dependencies**: Calls other update scripts as needed

### `update_player_statistics.py`
- **Purpose**: Updates individual player statistics (batting, pitching, fielding)
- **Data Sources**: MLB Stats API
- **Updates**: Player performance metrics, season totals, recent game stats

### `update_teams_and_games.py`
- **Purpose**: Updates team information and game schedules/results
- **Data Sources**: MLB Stats API
- **Updates**: Team rosters, game outcomes, standings

### `update_game_data.py`
- **Purpose**: Updates detailed game-level data and box scores
- **Data Sources**: MLB Stats API  
- **Updates**: Individual game statistics, play-by-play data

## Automation

The main update script (`daily_database_update.py`) is automatically executed via cron job:
```bash
# Runs daily at 1:00 AM
0 1 * * * /Users/om/Desktop/propbettingv2/MLBPropModel/scripts/run_daily_update.sh
```

## Logs

All update activities are logged to `../logs/daily_update.log` for monitoring and debugging purposes. 