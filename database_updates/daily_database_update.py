#!/usr/bin/env python3
"""
MLB Database Update Script
--------------------------
This script updates the MLB prop betting database with the latest teams, players, stats, and game data.
It follows the approach used in the legacy scripts but ensures all data is up-to-date.
"""

import requests
import psycopg2
import logging
import time
import configparser
import os
import random
from datetime import datetime, timedelta
from psycopg2.extras import execute_values

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_database_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mlb_database_update')

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
        # Set autocommit to True to avoid transaction issues
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def verify_db_connection(config):
    """Verify database connection is working"""
    logger.info("Verifying database connection...")
    conn = connect_to_db(config)
    if not conn:
        logger.error("Failed to connect to database. Please check your database configuration.")
        return False
    conn.close()
    logger.info("Database connection successful!")
    return True

def fetch_mlb_teams(config):
    """Fetch all MLB teams and update the database"""
    logger.info("Fetching MLB teams...")
    
    url = f"{config['api_base_url']}/teams"
    params = {
        "sportId": 1,  # MLB
        "season": datetime.now().year
    }
    
    conn = connect_to_db(config)
    if not conn:
        return 0
    
    cur = conn.cursor()
    teams_added = 0
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for team in data.get('teams', []):
            try:
                team_id = team.get('id')
                name = team.get('name')
                abbreviation = team.get('abbreviation')
                team_code = team.get('teamCode')
                league_id = team.get('league', {}).get('id')
                league_name = team.get('league', {}).get('name')
                division_id = team.get('division', {}).get('id')
                division_name = team.get('division', {}).get('name')
                venue_id = team.get('venue', {}).get('id')
                venue_name = team.get('venue', {}).get('name')
                
                # Insert or update team
                cur.execute("""
                    INSERT INTO teams (team_id, name, abbreviation, team_code, league_id, league_name, 
                                    division_id, division_name, venue_id, venue_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (team_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        abbreviation = EXCLUDED.abbreviation,
                        team_code = EXCLUDED.team_code,
                        league_id = EXCLUDED.league_id,
                        league_name = EXCLUDED.league_name,
                        division_id = EXCLUDED.division_id,
                        division_name = EXCLUDED.division_name,
                        venue_id = EXCLUDED.venue_id,
                        venue_name = EXCLUDED.venue_name
                """, (team_id, name, abbreviation, team_code, league_id, league_name, 
                    division_id, division_name, venue_id, venue_name))
                teams_added += 1
            except Exception as e:
                logger.error(f"Error processing team {team.get('name', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully added/updated {teams_added} teams")
        return teams_added
    
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return 0
    finally:
        cur.close()
        conn.close()

def fetch_players(config, team_id=None):
    """Fetch players for all teams or a specific team"""
    logger.info(f"Fetching players for {'all teams' if team_id is None else f'team {team_id}'}...")
    
    conn = connect_to_db(config)
    if not conn:
        return 0
    
    cur = conn.cursor()
    players_added = 0
    
    try:
        # Get teams to process
        if team_id:
            cur.execute("SELECT team_id FROM teams WHERE team_id = %s", (team_id,))
        else:
            cur.execute("SELECT team_id FROM teams")
        
        teams = cur.fetchall()
        
        for team in teams:
            team_id = team[0]
            logger.info(f"Fetching players for team ID {team_id}")
            
            url = f"{config['api_base_url']}/teams/{team_id}/roster"
            params = {
                "rosterType": "active",
                "season": datetime.now().year,
                "hydrate": "person"
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for player in data.get('roster', []):
                    person = player.get('person', {})
                    position = player.get('position', {})
                    
                    player_id = person.get('id')
                    first_name = person.get('firstName', '')
                    last_name = person.get('lastName', '')
                    full_name = person.get('fullName', '')
                    jersey_number = player.get('jerseyNumber', '')
                    position_code = position.get('abbreviation', '')
                    position_name = position.get('name', '')
                    position_type = position.get('type', '')
                    bats = person.get('batSide', {}).get('code', '')
                    throws = person.get('pitchHand', {}).get('code', '')
                    
                    # Insert or update player
                    cur.execute("""
                        INSERT INTO players (player_id, team_id, first_name, last_name, full_name, 
                                           jersey_number, position, position_name, position_type, bats, throws)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id) DO UPDATE SET
                            team_id = EXCLUDED.team_id,
                            first_name = EXCLUDED.first_name,
                            last_name = EXCLUDED.last_name,
                            full_name = EXCLUDED.full_name,
                            jersey_number = EXCLUDED.jersey_number,
                            position = EXCLUDED.position,
                            position_name = EXCLUDED.position_name,
                            position_type = EXCLUDED.position_type,
                            bats = EXCLUDED.bats,
                            throws = EXCLUDED.throws
                    """, (player_id, team_id, first_name, last_name, full_name, 
                          jersey_number, position_code, position_name, position_type, bats, throws))
                    players_added += 1
                
                conn.commit()
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching players for team {team_id}: {e}")
                conn.rollback()
        
        logger.info(f"Successfully added/updated {players_added} players")
        return players_added
    
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def fetch_player_stats(config, days_back=30, batch_size=20, delay_between_batches=3):
    """Fetch recent player statistics with batching for better reliability"""
    logger.info(f"Fetching player statistics for the last {days_back} days in batches of {batch_size}...")

    conn = connect_to_db(config)
    if not conn:
        return 0

    cur = conn.cursor()
    stats_added = 0

    try:
        # Get all players from database
        cur.execute("SELECT player_id, first_name, last_name FROM players")
        all_players = cur.fetchall()

        # Shuffle players to distribute the load more evenly
        random.shuffle(all_players)

        total_players = len(all_players)
        logger.info(f"Found {total_players} players to process")

        batches = [all_players[i:i + batch_size] for i in range(0, total_players, batch_size)]

        for batch_num, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_num+1}/{len(batches)} ({len(batch)} players)")
            batch_stats_added = 0

            for player in batch:
                player_id = player[0]
                player_name = f"{player[1]} {player[2]}"

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
                                    # Insert batting stats
                                    batting_sql = """
                                        INSERT INTO player_batting_stats (
                                            player_id, first_name, last_name, date, stat_period, season,
                                            games_played, at_bats, runs, hits, doubles, triples, home_runs,
                                            rbis, walks, strikeouts, stolen_bases, batting_avg, obp, slg, ops,
                                            last_updated
                                        ) VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                                        )
                                        ON CONFLICT (player_id, season, stat_period) 
                                        DO UPDATE SET
                                            first_name = EXCLUDED.first_name,
                                            last_name = EXCLUDED.last_name,
                                            date = EXCLUDED.date,
                                            games_played = EXCLUDED.games_played,
                                            at_bats = EXCLUDED.at_bats,
                                            runs = EXCLUDED.runs,
                                            hits = EXCLUDED.hits,
                                            doubles = EXCLUDED.doubles,
                                            triples = EXCLUDED.triples,
                                            home_runs = EXCLUDED.home_runs,
                                            rbis = EXCLUDED.rbis,
                                            walks = EXCLUDED.walks,
                                            strikeouts = EXCLUDED.strikeouts,
                                            stolen_bases = EXCLUDED.stolen_bases,
                                            batting_avg = EXCLUDED.batting_avg,
                                            obp = EXCLUDED.obp,
                                            slg = EXCLUDED.slg,
                                            ops = EXCLUDED.ops,
                                            last_updated = CURRENT_TIMESTAMP
                                    """
                                    cur.execute(batting_sql, (
                                        player_id, player[1], player[2], datetime.now().date(), 'season', datetime.now().year,
                                        stat_data.get('gamesPlayed'),
                                        stat_data.get('atBats'),
                                        stat_data.get('runs'),
                                        stat_data.get('hits'),
                                        stat_data.get('doubles'),
                                        stat_data.get('triples'),
                                        stat_data.get('homeRuns'),
                                        stat_data.get('rbi'),
                                        stat_data.get('baseOnBalls'),
                                        stat_data.get('strikeOuts'),
                                        stat_data.get('stolenBases'),
                                        stat_data.get('avg'),
                                        stat_data.get('obp'),
                                        stat_data.get('slg'),
                                        stat_data.get('ops')
                                    ))
                                    batch_stats_added += 1
                                
                                elif group_name == 'pitching':
                                    # Insert pitching stats
                                    pitching_sql = """
                                        INSERT INTO player_pitching_stats (
                                            player_id, first_name, last_name, date, stat_period, season,
                                            games_played, games_started, wins, losses, saves, innings_pitched,
                                            hits_allowed, runs_allowed, earned_runs, walks_allowed, strikeouts,
                                            era, whip, batting_avg_against, last_updated
                                        ) VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                                        )
                                        ON CONFLICT (player_id, season, stat_period)
                                        DO UPDATE SET
                                            first_name = EXCLUDED.first_name,
                                            last_name = EXCLUDED.last_name,
                                            date = EXCLUDED.date,
                                            games_played = EXCLUDED.games_played,
                                            games_started = EXCLUDED.games_started,
                                            wins = EXCLUDED.wins,
                                            losses = EXCLUDED.losses,
                                            saves = EXCLUDED.saves,
                                            innings_pitched = EXCLUDED.innings_pitched,
                                            hits_allowed = EXCLUDED.hits_allowed,
                                            runs_allowed = EXCLUDED.runs_allowed,
                                            earned_runs = EXCLUDED.earned_runs,
                                            walks_allowed = EXCLUDED.walks_allowed,
                                            strikeouts = EXCLUDED.strikeouts,
                                            era = EXCLUDED.era,
                                            whip = EXCLUDED.whip,
                                            batting_avg_against = EXCLUDED.batting_avg_against,
                                            last_updated = CURRENT_TIMESTAMP
                                    """
                                    cur.execute(pitching_sql, (
                                        player_id, player[1], player[2], datetime.now().date(), 'season', datetime.now().year,
                                        stat_data.get('gamesPlayed'),
                                        stat_data.get('gamesStarted'),
                                        stat_data.get('wins'),
                                        stat_data.get('losses'),
                                        stat_data.get('saves'),
                                        stat_data.get('inningsPitched'),
                                        stat_data.get('hits'),
                                        stat_data.get('runs'),
                                        stat_data.get('earnedRuns'),
                                        stat_data.get('baseOnBalls'),
                                        stat_data.get('strikeOuts'),
                                        stat_data.get('era'),
                                        stat_data.get('whip'),
                                        stat_data.get('avg')
                                    ))
                                    batch_stats_added += 1

                except Exception as e:
                    logger.error(f"Error fetching stats for player {player_id} ({player_name}): {e}")
                    continue  # Continue with next player even if this one fails

            conn.commit()
            logger.info(f"Added/updated {batch_stats_added} stats in batch {batch_num+1}")
            stats_added += batch_stats_added

            # Add a delay between batches to avoid hitting rate limits
            if batch_num < len(batches) - 1:
                logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)

        logger.info(f"Successfully added/updated {stats_added} player statistics")
        return stats_added

    except Exception as e:
        logger.error(f"Error fetching player statistics: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def fetch_game_schedule(config, days_forward=14, days_back=30):
    """Fetch MLB game schedule and results with an extended date range"""
    logger.info(f"Fetching game schedule ({days_back} days back, {days_forward} days forward)...")

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_forward)).strftime("%Y-%m-%d")

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

                # Insert or update game
                cur.execute("""
                    INSERT INTO games (
                        game_id, date, home_team_id, away_team_id, venue, venue_is_dome, game_time, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id) DO UPDATE SET
                        date = EXCLUDED.date,
                        home_team_id = EXCLUDED.home_team_id,
                        away_team_id = EXCLUDED.away_team_id,
                        venue = EXCLUDED.venue,
                        venue_is_dome = EXCLUDED.venue_is_dome,
                        game_time = EXCLUDED.game_time,
                        status = EXCLUDED.status
                """, (
                    game_id, game_date, home_team_id, away_team_id, venue, venue_is_dome, game_time, status
                ))
                games_added += 1

        conn.commit()
        logger.info(f"Successfully added/updated {games_added} games")
        return games_added

    except Exception as e:
        logger.error(f"Error fetching game schedule: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def fetch_per_game_stats(config, days_back=30):
    """Fetch per-game stats for all players for the last N days and insert into stats tables."""
    logger.info(f"Fetching per-game stats for all games in the last {days_back} days...")
    conn = connect_to_db(config)
    if not conn:
        return 0
    cur = conn.cursor()
    stats_added = 0
    try:
        # Get all games in the last N days
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        cur.execute("SELECT game_id, date FROM games WHERE date >= %s", (start_date,))
        games = cur.fetchall()
        logger.info(f"Found {len(games)} games to process.")
        for game_id, game_date in games:
            url = f"{config['api_base_url']}/game/{game_id}/boxscore"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                # Batting stats
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
                                    player_id, date, season, games, at_bats, hits, runs, doubles, triples, home_runs, rbi, stolen_bases, caught_stealing, base_on_balls, strikeouts, batting_average, on_base_percentage, slugging_percentage, ops, stat_period, last_updated
                                ) VALUES (
                                    %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'game', NOW()
                                ) ON CONFLICT (player_id, date, stat_period) DO UPDATE SET
                                    at_bats = EXCLUDED.at_bats,
                                    hits = EXCLUDED.hits,
                                    runs = EXCLUDED.runs,
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
                                pid, game_date, datetime.now().year,
                                stats.get('atBats'), stats.get('hits'), stats.get('runs'), stats.get('doubles'), stats.get('triples'), stats.get('homeRuns'), stats.get('rbi'), stats.get('stolenBases'), stats.get('caughtStealing'), stats.get('baseOnBalls'), stats.get('strikeOuts'), stats.get('avg'), stats.get('obp'), stats.get('slg'), stats.get('ops')
                            ))
                            stats_added += 1
                # Pitching stats
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
                                    player_id, date, season, wins, losses, games, games_started, saves, innings_pitched, hits_allowed, runs_allowed, earned_runs, walks_allowed, strikeouts, era, whip, stat_period, last_updated
                                ) VALUES (
                                    %s, %s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'game', NOW()
                                ) ON CONFLICT (player_id, date, stat_period) DO UPDATE SET
                                    wins = EXCLUDED.wins,
                                    losses = EXCLUDED.losses,
                                    games_started = EXCLUDED.games_started,
                                    saves = EXCLUDED.saves,
                                    innings_pitched = EXCLUDED.innings_pitched,
                                    hits_allowed = EXCLUDED.hits_allowed,
                                    runs_allowed = EXCLUDED.runs_allowed,
                                    earned_runs = EXCLUDED.earned_runs,
                                    walks_allowed = EXCLUDED.walks_allowed,
                                    strikeouts = EXCLUDED.strikeouts,
                                    era = EXCLUDED.era,
                                    whip = EXCLUDED.whip,
                                    last_updated = NOW()
                            """, (
                                pid, game_date, datetime.now().year,
                                stats.get('wins'), stats.get('losses'), stats.get('gamesStarted'), stats.get('saves'), stats.get('inningsPitched'), stats.get('hits'), stats.get('runs'), stats.get('earnedRuns'), stats.get('baseOnBalls'), stats.get('strikeOuts'), stats.get('era'), stats.get('whip')
                            ))
                            stats_added += 1
            except Exception as e:
                logger.error(f"Error fetching boxscore for game {game_id}: {e}")
                continue
        conn.commit()
        logger.info(f"Successfully added/updated {stats_added} per-game player stats.")
        return stats_added
    except Exception as e:
        logger.error(f"Error in fetch_per_game_stats: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()

def update_matchups(config):
    """Update the matchups table with pitcher-batter matchups"""
    logger.info("Updating matchups table...")
    
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

def verify_data_integrity(config):
    """Verify that data was properly restored by checking counts"""
    logger.info("Verifying data integrity...")
    
    conn = connect_to_db(config)
    if not conn:
        return False
    
    cur = conn.cursor()
    
    try:
        # Check teams
        cur.execute("SELECT COUNT(*) FROM teams")
        team_count = cur.fetchone()[0]
        logger.info(f"Teams in database: {team_count}")
        
        # Check players
        cur.execute("SELECT COUNT(*) FROM players")
        player_count = cur.fetchone()[0]
        logger.info(f"Players in database: {player_count}")
        
        # Check player batting stats
        cur.execute("SELECT COUNT(*) FROM player_batting_stats")
        batting_stats_count = cur.fetchone()[0]
        logger.info(f"Player batting stats in database: {batting_stats_count}")
        
        # Check player pitching stats
        cur.execute("SELECT COUNT(*) FROM player_pitching_stats")
        pitching_stats_count = cur.fetchone()[0]
        logger.info(f"Player pitching stats in database: {pitching_stats_count}")
        
        # Check games
        cur.execute("SELECT COUNT(*) FROM games")
        game_count = cur.fetchone()[0]
        logger.info(f"Games in database: {game_count}")
        
        # Check matchups
        cur.execute("SELECT COUNT(*) FROM matchups")
        matchup_count = cur.fetchone()[0]
        logger.info(f"Matchups in database: {matchup_count}")
        
        # Check most recent game date
        cur.execute("SELECT MAX(date) FROM games")
        most_recent_game = cur.fetchone()[0]
        logger.info(f"Most recent game date in database: {most_recent_game}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error verifying data integrity: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def update_mlb_database():
    """Update the MLB database with the latest data"""
    logger.info("Starting MLB database update process")
    
    # Load configuration
    config = load_config()
    
    # Verify database connection
    if not verify_db_connection(config):
        return False
    
    # Fetch and update teams
    logger.info("Step 1: Updating teams...")
    teams_added = fetch_mlb_teams(config)
    logger.info(f"Added/updated {teams_added} teams")
    
    # Fetch and update players
    logger.info("Step 2: Updating players...")
    players_added = fetch_players(config)
    logger.info(f"Added/updated {players_added} players")
    
    # Fetch and update player statistics
    logger.info("Step 3: Updating player statistics...")
    stats_added = fetch_player_stats(config, days_back=30)
    logger.info(f"Added/updated {stats_added} player statistics")
    
    # Fetch and update game schedule with extended date range
    logger.info("Step 4: Updating game schedule...")
    games_added = fetch_game_schedule(config, days_forward=14, days_back=30)
    logger.info(f"Added/updated {games_added} games")
    
    # Fetch and update per-game stats for all players for the last 30 days
    logger.info("Step 5: Updating per-game player stats...")
    per_game_stats_added = fetch_per_game_stats(config, days_back=30)
    logger.info(f"Added/updated {per_game_stats_added} per-game player stats")
    
    # Update matchups
    logger.info("Step 6: Updating matchups...")
    matchups_added = update_matchups(config)
    logger.info(f"Added {matchups_added} matchups")
    
    # Verify data integrity
    logger.info("Step 7: Verifying data integrity...")
    verify_data_integrity(config)
    
    logger.info("MLB database update completed successfully!")
    return True

if __name__ == "__main__":
    update_mlb_database()
