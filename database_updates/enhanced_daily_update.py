#!/usr/bin/env python3
"""
Enhanced MLB Database Update Script
File: database_updates/enhanced_daily_update.py
Handles connection issues and prevents overpopulation
"""
import requests
import psycopg2
import logging
import time
import configparser
import os
import sys
from datetime import datetime, timedelta

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/enhanced_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_mlb_update')

def load_config():
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return {
        'db_host': config.get('Database', 'host', fallback='localhost'),
        'db_name': config.get('Database', 'database', fallback='mlb_prop_betting'),
        'db_user': config.get('Database', 'user', fallback='postgres'),
        'db_password': config.get('Database', 'password', fallback='ophadke'),
        'api_base_url': config.get('API', 'base_url', fallback='https://statsapi.mlb.com/api/v1'),
    }

def connect_to_db_with_retry(config, max_retries=3):
    """Connect to database with retry logic"""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=config['db_host'],
                database=config['db_name'],
                user=config['db_user'],
                password=config['db_password'],
                connect_timeout=15
            )
            logger.info("Database connection successful")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                logger.error("All connection attempts failed")
                return None

def get_latest_db_date(config):
    """Get the latest date in the database to avoid overpopulation"""
    conn = connect_to_db_with_retry(config)
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM games")
        latest_date = cur.fetchone()[0]
        conn.close()
        return latest_date
    except Exception as e:
        logger.error(f"Error getting latest date: {e}")
        return None

def update_recent_games_only(config):
    """Update only games from the last few days to prevent overpopulation"""
    logger.info("Updating recent games...")
    
    # Get latest date in DB
    latest_db_date = get_latest_db_date(config)
    
    if latest_db_date:
        # Start from latest date in DB, go forward to today + 5 days
        start_date = latest_db_date
        end_date = (datetime.now() + timedelta(days=5)).date()
        logger.info(f"Updating games from {start_date} to {end_date}")
    else:
        # If no data, get last 7 days
        start_date = (datetime.now() - timedelta(days=7)).date()
        end_date = (datetime.now() + timedelta(days=2)).date()
        logger.info(f"No existing data found. Getting games from {start_date} to {end_date}")
    
    # Fix the date logic - make sure start_date is not after end_date
    if start_date > end_date:
        # If latest DB date is in the future, get recent past data instead
        start_date = (datetime.now() - timedelta(days=3)).date()
        end_date = datetime.now().date()
        logger.info(f"Adjusted date range to: {start_date} to {end_date}")
    
    url = f"{config['api_base_url']}/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "gameType": "R",
        "hydrate": "team,venue"
    }
    
    conn = connect_to_db_with_retry(config)
    if not conn:
        return 0
    
    cur = conn.cursor()
    games_processed = 0
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if "dates" not in data:
            logger.info("No game data available for the specified period")
            return 0
        
        for date in data.get('dates', []):
            for game in date.get('games', []):
                game_id = game.get('gamePk')
                game_date = date.get('date')
                status = game.get('status', {}).get('detailedState')
                
                home_team = game.get('teams', {}).get('home', {}).get('team', {})
                away_team = game.get('teams', {}).get('away', {}).get('team', {})
                
                home_team_id = home_team.get('id')
                away_team_id = away_team.get('id')
                venue = game.get('venue', {}).get('name')
                
                # Insert or update game
                cur.execute("""
                    INSERT INTO games (
                        game_id, date, home_team_id, away_team_id, venue, status
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        venue = EXCLUDED.venue
                """, (game_id, game_date, home_team_id, away_team_id, venue, status))
                
                games_processed += 1
        
        conn.commit()
        logger.info(f"Successfully processed {games_processed} games")
        return games_processed
        
    except Exception as e:
        logger.error(f"Error updating games: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def update_recent_stats_only(config):
    """Update stats only for recent completed games"""
    logger.info("Updating recent player stats...")
    
    conn = connect_to_db_with_retry(config)
    if not conn:
        return 0
    
    cur = conn.cursor()
    stats_added = 0
    
    try:
        # Get recent completed games that don't have stats yet
        cur.execute("""
            SELECT g.game_id, g.date 
            FROM games g
            LEFT JOIN player_batting_stats pbs ON g.date = pbs.date AND pbs.stat_period = 'game'
            WHERE g.date >= CURRENT_DATE - INTERVAL '5 days'
              AND g.status IN ('Final', 'Game Over')
              AND pbs.date IS NULL
            ORDER BY g.date DESC
            LIMIT 20
        """)
        
        games_needing_stats = cur.fetchall()
        logger.info(f"Found {len(games_needing_stats)} games needing stats")
        
        for game_id, game_date in games_needing_stats:
            logger.info(f"Processing stats for game {game_id} on {game_date}")
            
            # Get box score
            url = f"{config['api_base_url']}/game/{game_id}/boxscore"
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Process batting stats for both teams
                for team_type in ['home', 'away']:
                    batters = data.get('teams', {}).get(team_type, {}).get('batters', [])
                    players = data.get('teams', {}).get(team_type, {}).get('players', {})
                    
                    for pid in batters:
                        player_key = f"ID{pid}"
                        player = players.get(player_key, {})
                        stats = player.get('stats', {}).get('batting', {})
                        
                        if stats:
                            cur.execute("""
                                INSERT INTO player_batting_stats (
                                    player_id, date, season, games, at_bats, hits, runs,
                                    doubles, triples, home_runs, rbi, stolen_bases,
                                    base_on_balls, strikeouts, stat_period, last_updated
                                ) VALUES (
                                    %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'game', NOW()
                                ) ON CONFLICT (player_id, date, stat_period) DO NOTHING
                            """, (
                                pid, game_date, datetime.now().year,
                                stats.get('atBats', 0), stats.get('hits', 0), stats.get('runs', 0),
                                stats.get('doubles', 0), stats.get('triples', 0), stats.get('homeRuns', 0),
                                stats.get('rbi', 0), stats.get('stolenBases', 0),
                                stats.get('baseOnBalls', 0), stats.get('strikeOuts', 0)
                            ))
                            stats_added += 1
                
                # Process pitching stats
                for team_type in ['home', 'away']:
                    pitchers = data.get('teams', {}).get(team_type, {}).get('pitchers', [])
                    players = data.get('teams', {}).get(team_type, {}).get('players', {})
                    
                    for pid in pitchers:
                        player_key = f"ID{pid}"
                        player = players.get(player_key, {})
                        stats = player.get('stats', {}).get('pitching', {})
                        
                        if stats:
                            cur.execute("""
                                INSERT INTO player_pitching_stats (
                                    player_id, date, season, games, innings_pitched,
                                    hits_allowed, runs_allowed, earned_runs, walks_allowed,
                                    strikeouts, stat_period, last_updated
                                ) VALUES (
                                    %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, 'game', NOW()
                                ) ON CONFLICT (player_id, date, stat_period) DO NOTHING
                            """, (
                                pid, game_date, datetime.now().year,
                                stats.get('inningsPitched', 0), stats.get('hits', 0),
                                stats.get('runs', 0), stats.get('earnedRuns', 0),
                                stats.get('baseOnBalls', 0), stats.get('strikeOuts', 0)
                            ))
                
                conn.commit()
                logger.info(f"Processed stats for game {game_id}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                continue
        
        logger.info(f"Added {stats_added} new stat records")
        return stats_added
        
    except Exception as e:
        logger.error(f"Error updating stats: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def main():
    """Main update function"""
    logger.info("Starting enhanced MLB database update")
    
    config = load_config()
    
    # Test connection first
    test_conn = connect_to_db_with_retry(config)
    if not test_conn:
        logger.error("Cannot connect to database. Exiting.")
        return False
    test_conn.close()
    
    # Update recent games
    games_updated = update_recent_games_only(config)
    
    # Update recent stats
    stats_updated = update_recent_stats_only(config)
    
    logger.info(f"Enhanced update complete: {games_updated} games, {stats_updated} stats")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 