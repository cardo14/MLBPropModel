#!/usr/bin/env python3
"""
MLB Player Statistics Update Script
----------------------------------
This script updates the player_batting_stats and player_pitching_stats tables in the MLB prop betting database
with the latest statistics from the MLB Stats API.
"""

import requests
import psycopg2
import logging
import configparser
import os
import random
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_stats_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mlb_stats_update')

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

def update_player_stats(config, batch_size=20, delay_between_batches=3):
    """Update player statistics in the database"""
    logger.info(f"Updating player statistics in batches of {batch_size}...")
    
    conn = connect_to_db(config)
    if not conn:
        return 0
    
    cur = conn.cursor()
    stats_updated = 0
    
    try:
        # Get all players from database
        cur.execute("SELECT player_id, first_name, last_name, position FROM players")
        all_players = cur.fetchall()
        
        # Shuffle players to distribute the load more evenly
        random.shuffle(all_players)
        
        total_players = len(all_players)
        logger.info(f"Found {total_players} players to process")
        
        batches = [all_players[i:i + batch_size] for i in range(0, total_players, batch_size)]
        
        for batch_num, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_num+1}/{len(batches)} ({len(batch)} players)")
            batch_stats_updated = 0
            
            for player in batch:
                player_id = player[0]
                first_name = player[1]
                last_name = player[2]
                position = player[3]
                player_name = f"{first_name} {last_name}"
                
                # MLB Stats API endpoint for player stats
                url = f"{config['api_base_url']}/people/{player_id}/stats"
                params = {
                    "stats": "season",
                    "season": str(datetime.now().year),
                    "group": "hitting,pitching",
                    "gameType": "R"    # Regular season
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    for stat_group in data.get('stats', []):
                        group_name = stat_group.get('group', {}).get('displayName', '').lower()
                        
                        if group_name in ['hitting', 'pitching']:
                            stats = stat_group.get('splits', [])
                            
                            if stats and len(stats) > 0:
                                stat_data = stats[0].get('stat', {})
                                
                                if group_name == 'hitting':
                                    # Insert or update hitting stats
                                    cur.execute("""
                                        INSERT INTO player_batting_stats (
                                            player_id, season, games, at_bats, runs, hits, doubles, triples,
                                            home_runs, rbi, stolen_bases, caught_stealing, base_on_balls,
                                            strikeouts, batting_average, on_base_percentage, 
                                            slugging_percentage, ops, last_updated
                                        )
                                        VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                                        )
                                        ON CONFLICT (player_id, season) DO UPDATE SET
                                            games = EXCLUDED.games,
                                            at_bats = EXCLUDED.at_bats,
                                            runs = EXCLUDED.runs,
                                            hits = EXCLUDED.hits,
                                            doubles = EXCLUDED.doubles,
                                            triples = EXCLUDED.triples,
                                            home_runs = EXCLUDED.home_runs,
                                            rbi = EXCLUDED.rbi,
                                            stolen_bases = EXCLUDED.stolen_bases,
                                            caught_stealing = EXCLUDED.caught_stealing,
                                            base_on_balls = EXCLUDED.base_on_balls,
                                            strikeouts = EXCLUDED.strikeouts,
                                            batting_average = EXCLUDED.batting_average,
                                            on_base_percentage = EXCLUDED.on_base_percentage,
                                            slugging_percentage = EXCLUDED.slugging_percentage,
                                            ops = EXCLUDED.ops,
                                            last_updated = NOW()
                                    """, (
                                        player_id, datetime.now().year,
                                        stat_data.get('gamesPlayed'),
                                        stat_data.get('atBats'),
                                        stat_data.get('runs'),
                                        stat_data.get('hits'),
                                        stat_data.get('doubles'),
                                        stat_data.get('triples'),
                                        stat_data.get('homeRuns'),
                                        stat_data.get('rbi'),
                                        stat_data.get('stolenBases'),
                                        stat_data.get('caughtStealing'),
                                        stat_data.get('baseOnBalls'),
                                        stat_data.get('strikeOuts'),
                                        stat_data.get('avg'),
                                        stat_data.get('obp'),
                                        stat_data.get('slg'),
                                        stat_data.get('ops')
                                    ))
                                    logger.info(f"Updated batting stats for {player_name} (ID: {player_id})")
                                    batch_stats_updated += 1
                                
                                elif group_name == 'pitching':
                                    # Insert or update pitching stats
                                    cur.execute("""
                                        INSERT INTO player_pitching_stats (
                                            player_id, season, wins, losses, era, games, games_started,
                                            saves, innings_pitched, hits_allowed, runs_allowed, earned_runs,
                                            walks_allowed, strikeouts, whip, last_updated
                                        )
                                        VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                                        )
                                        ON CONFLICT (player_id, season) DO UPDATE SET
                                            wins = EXCLUDED.wins,
                                            losses = EXCLUDED.losses,
                                            era = EXCLUDED.era,
                                            games = EXCLUDED.games,
                                            games_started = EXCLUDED.games_started,
                                            saves = EXCLUDED.saves,
                                            innings_pitched = EXCLUDED.innings_pitched,
                                            hits_allowed = EXCLUDED.hits_allowed,
                                            runs_allowed = EXCLUDED.runs_allowed,
                                            earned_runs = EXCLUDED.earned_runs,
                                            walks_allowed = EXCLUDED.walks_allowed,
                                            strikeouts = EXCLUDED.strikeouts,
                                            whip = EXCLUDED.whip,
                                            last_updated = NOW()
                                    """, (
                                        player_id, datetime.now().year,
                                        stat_data.get('wins'),
                                        stat_data.get('losses'),
                                        stat_data.get('era'),
                                        stat_data.get('gamesPlayed'),
                                        stat_data.get('gamesStarted'),
                                        stat_data.get('saves'),
                                        stat_data.get('inningsPitched'),
                                        stat_data.get('hits'),
                                        stat_data.get('runs'),
                                        stat_data.get('earnedRuns'),
                                        stat_data.get('baseOnBalls'),
                                        stat_data.get('strikeOuts'),
                                        stat_data.get('whip')
                                    ))
                                    logger.info(f"Updated pitching stats for {player_name} (ID: {player_id})")
                                    batch_stats_updated += 1
                
                except Exception as e:
                    logger.error(f"Error fetching stats for player {player_id} ({player_name}): {e}")
                    continue  # Continue with next player even if this one fails
            
            conn.commit()
            logger.info(f"Updated {batch_stats_updated} stats in batch {batch_num+1}")
            stats_updated += batch_stats_updated
            
            # Add a delay between batches to avoid hitting rate limits
            if batch_num < len(batches) - 1:
                logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)
        
        logger.info(f"Successfully updated {stats_updated} player statistics")
        
        # Check the most recent update timestamp in the database
        cur.execute("SELECT MAX(last_updated) FROM player_batting_stats")
        most_recent_batting = cur.fetchone()[0]
        
        cur.execute("SELECT MAX(last_updated) FROM player_pitching_stats")
        most_recent_pitching = cur.fetchone()[0]
        
        logger.info(f"Most recent batting stats update: {most_recent_batting}")
        logger.info(f"Most recent pitching stats update: {most_recent_pitching}")
        
        return stats_updated
    
    except Exception as e:
        logger.error(f"Error updating player statistics: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def update_matchups(config):
    """Update matchups for upcoming games"""
    logger.info("Updating matchups for upcoming games...")
    
    conn = connect_to_db(config)
    if not conn:
        return 0
    
    cur = conn.cursor()
    matchups_added = 0
    
    try:
        # Get upcoming games
        cur.execute("""
            SELECT game_id, home_team_id, away_team_id, date
            FROM games
            WHERE date >= CURRENT_DATE
            ORDER BY date ASC
        """)
        games = cur.fetchall()
        
        for game in games:
            game_id, home_team_id, away_team_id, game_date = game
            
            # Get home team pitchers
            cur.execute("""
                SELECT player_id
                FROM players
                WHERE team_id = %s AND position = 'P'
            """, (home_team_id,))
            home_pitchers = cur.fetchall()
            
            # Get away team pitchers
            cur.execute("""
                SELECT player_id
                FROM players
                WHERE team_id = %s AND position = 'P'
            """, (away_team_id,))
            away_pitchers = cur.fetchall()
            
            # Get home team batters
            cur.execute("""
                SELECT player_id
                FROM players
                WHERE team_id = %s AND position != 'P'
            """, (home_team_id,))
            home_batters = cur.fetchall()
            
            # Get away team batters
            cur.execute("""
                SELECT player_id
                FROM players
                WHERE team_id = %s AND position != 'P'
            """, (away_team_id,))
            away_batters = cur.fetchall()
            
            # Create matchups for home batters vs away pitchers
            for batter in home_batters:
                for pitcher in away_pitchers:
                    # Check if matchup already exists
                    cur.execute("""
                        SELECT matchup_id FROM matchups 
                        WHERE game_id = %s AND pitching_id = %s AND batting_id = %s
                    """, (game_id, pitcher[0], batter[0]))
                    existing = cur.fetchone()
                    
                    if existing:
                        # Update existing matchup
                        cur.execute("""
                            UPDATE matchups 
                            SET updated_at = NOW()
                            WHERE matchup_id = %s
                        """, (existing[0],))
                    else:
                        # Insert new matchup
                        cur.execute("""
                            INSERT INTO matchups (
                                game_id, pitching_id, batting_id, 
                                batting_walk_recorded, batting_strikeout_recorded, 
                                updated_at
                            )
                            VALUES (%s, %s, %s, 0, 0, NOW())
                        """, (game_id, pitcher[0], batter[0]))
                        matchups_added += 1
            
            # Create matchups for away batters vs home pitchers
            for batter in away_batters:
                for pitcher in home_pitchers:
                    # Check if matchup already exists
                    cur.execute("""
                        SELECT matchup_id FROM matchups 
                        WHERE game_id = %s AND pitching_id = %s AND batting_id = %s
                    """, (game_id, pitcher[0], batter[0]))
                    existing = cur.fetchone()
                    
                    if existing:
                        # Update existing matchup
                        cur.execute("""
                            UPDATE matchups 
                            SET updated_at = NOW()
                            WHERE matchup_id = %s
                        """, (existing[0],))
                    else:
                        # Insert new matchup
                        cur.execute("""
                            INSERT INTO matchups (
                                game_id, pitching_id, batting_id, 
                                batting_walk_recorded, batting_strikeout_recorded, 
                                updated_at
                            )
                            VALUES (%s, %s, %s, 0, 0, NOW())
                        """, (game_id, pitcher[0], batter[0]))
                        matchups_added += 1
        
        conn.commit()
        logger.info(f"Successfully added {matchups_added} matchups")
        return matchups_added
    
    except Exception as e:
        logger.error(f"Error updating matchups: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to update player statistics and matchups"""
    logger.info("Starting MLB player statistics update")
    
    # Load configuration
    config = load_config()
    
    # Update player statistics
    stats_updated = update_player_stats(config)
    logger.info(f"Updated {stats_updated} player statistics")
    
    # Update matchups for upcoming games
    matchups_added = update_matchups(config)
    logger.info(f"Added {matchups_added} matchups")
    
    logger.info("MLB player statistics update completed")

if __name__ == "__main__":
    main()
