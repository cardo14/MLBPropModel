#!/usr/bin/env python3
"""
MLB Games Update Script
-----------------------
This script updates the games table in the MLB prop betting database with the latest game data.
It focuses specifically on getting the most recent games.
"""

import requests
import psycopg2
import logging
import configparser
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_games_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mlb_games_update')

def load_config():
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    if os.path.exists('config.ini'):
        config.read('config.ini')
        return {
            'db_host': config.get('Database', 'host', fallback='localhost'),
            'db_name': config.get('Database', 'database', fallback='mlb_prop_betting'),
            'db_user': config.get('Database', 'user', fallback='postgres'),
            'db_password': config.get('Database', 'password', fallback='ophadke'),
            'api_base_url': config.get('API', 'base_url', fallback='https://statsapi.mlb.com/api/v1'),
        }
    return {
        'db_host': 'localhost',
        'db_name': 'mlb_prop_betting',
        'db_user': 'postgres',
        'db_password': 'ophadke',
        'api_base_url': 'https://statsapi.mlb.com/api/v1',
    }

def connect_to_db(config):
    """Connect to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=config['db_host'],
            database=config['db_name'],
            user=config['db_user'],
            password=config['db_password']
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def fetch_game_schedule(config, days_forward=14, days_back=30):
    """Fetch MLB game schedule and results with an extended date range"""
    logger.info(f"Fetching game schedule ({days_back} days back, {days_forward} days forward)...")

    # Calculate date range
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_forward)).strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {start_date} to {end_date}")

    url = f"{config['api_base_url']}/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",
        "hydrate": "team,venue,weather"
    }

    conn = connect_to_db(config)
    if not conn:
        return 0

    cur = conn.cursor()
    games_added = 0

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "dates" not in data:
            logger.error("No game data returned from API")
            return 0

        for date in data.get('dates', []):
            logger.info(f"Processing games for date: {date.get('date')}")
            for game in date.get('games', []):
                game_id = game.get('gamePk')
                game_date = date.get('date')
                status = game.get('status', {}).get('detailedState')
                
                home_team = game.get('teams', {}).get('home', {}).get('team', {})
                away_team = game.get('teams', {}).get('away', {}).get('team', {})
                
                home_team_id = home_team.get('id')
                away_team_id = away_team.get('id')
                
                venue = game.get('venue', {}).get('name')
                
                # Check if venue is a dome
                venue_is_dome = False
                if venue in ['Minute Maid Park', 'Tropicana Field', 'Rogers Centre', 'T-Mobile Park', 'Globe Life Field', 'Miller Park', 'Chase Field']:
                    venue_is_dome = True
                
                # Game time (extract from gameDate)
                game_time = None
                if game.get('gameDate'):
                    try:
                        game_datetime = datetime.fromisoformat(game.get('gameDate').replace('Z', '+00:00'))
                        game_time = game_datetime.strftime('%H:%M')
                    except Exception as e:
                        logger.warning(f"Error parsing game time: {e}")

                # Check if game already exists
                cur.execute("SELECT game_id FROM games WHERE game_id = %s", (game_id,))
                existing_game = cur.fetchone()

                if existing_game:
                    # Update existing game
                    cur.execute("""
                        UPDATE games SET
                            date = %s,
                            home_team_id = %s,
                            away_team_id = %s,
                            venue = %s,
                            venue_is_dome = %s,
                            game_time = %s,
                            status = %s
                        WHERE game_id = %s
                    """, (
                        game_date, home_team_id, away_team_id, venue, 
                        venue_is_dome, game_time, status, game_id
                    ))
                    logger.info(f"Updated game {game_id} on {game_date}")
                else:
                    # Insert new game
                    cur.execute("""
                        INSERT INTO games (
                            game_id, date, home_team_id, away_team_id, 
                            venue, venue_is_dome, game_time, status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        game_id, game_date, home_team_id, away_team_id,
                        venue, venue_is_dome, game_time, status
                    ))
                    logger.info(f"Added new game {game_id} on {game_date}")
                
                games_added += 1

        conn.commit()
        logger.info(f"Successfully processed {games_added} games")
        
        # Check the most recent game date in the database
        cur.execute("SELECT MAX(date) FROM games")
        most_recent_date = cur.fetchone()[0]
        logger.info(f"Most recent game date in database: {most_recent_date}")
        
        return games_added

    except Exception as e:
        logger.error(f"Error fetching game schedule: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to update MLB games"""
    logger.info("Starting MLB games update")
    
    # Load configuration
    config = load_config()
    
    # Fetch and update game schedule with extended date range
    games_added = fetch_game_schedule(config, days_forward=14, days_back=30)
    logger.info(f"Processed {games_added} games")
    
    logger.info("MLB games update completed")

if __name__ == "__main__":
    main()
