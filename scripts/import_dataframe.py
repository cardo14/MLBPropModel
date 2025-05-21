#!/usr/bin/env python3
"""
Import MLB prop betting data from a CSV dataframe into the PostgreSQL database.
This script parses the data and distributes it across the appropriate tables.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.config import load_config, logger
from utils.db_utils import connect_to_db, reset_tables

def import_dataframe(config, csv_path, reset_db=False):
    """
    Import data from a CSV dataframe into the PostgreSQL database
    
    Args:
        config: Configuration dictionary
        csv_path: Path to the CSV file
        reset_db: Whether to reset the database tables before importing
    
    Returns:
        bool: Success or failure
    """
    logger.info(f"Importing data from {csv_path}...")
    
    # Connect to database
    conn = connect_to_db(config)
    if not conn:
        return False
    
    try:
        # Reset tables if requested
        if reset_db:
            logger.warning("Resetting all database tables!")
            if not reset_tables(config):
                logger.error("Failed to reset tables")
                return False
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataframe with {len(df)} rows")
        
        # Process the data
        process_teams_and_players(conn, df)
        process_games(conn, df)
        process_player_stats(conn, df)
        process_props_and_odds(conn, df)
        
        logger.info("Data import completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error importing data: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def process_teams_and_players(conn, df):
    """Process and import team and player data"""
    logger.info("Processing teams and players...")
    
    cur = conn.cursor()
    teams_added = 0
    players_added = 0
    
    try:
        # Extract unique pitchers
        pitchers = df[['pitching_id', 'pitching_name']].drop_duplicates()
        
        # Extract unique batters
        batters = df[['batting_id', 'batting_name']].drop_duplicates()
        
        # Combine all players (this is simplified - in reality we'd need team affiliations)
        all_players = []
        
        for _, row in pitchers.iterrows():
            all_players.append({
                'player_id': int(row['pitching_id']),
                'name': row['pitching_name'],
                'position': 'P',  # Assuming all pitchers have position P
                'is_pitcher': True
            })
        
        for _, row in batters.iterrows():
            # Skip if player is already in the list (might be both pitcher and batter)
            if any(p['player_id'] == int(row['batting_id']) for p in all_players):
                continue
                
            all_players.append({
                'player_id': int(row['batting_id']),
                'name': row['batting_name'],
                'position': 'Unknown',  # We don't have position data in the example
                'is_pitcher': False
            })
        
        # Extract game data to get teams
        games = df[['game_id', 'game_venue']].drop_duplicates()
        
        # For this example, we'll create dummy teams since we don't have explicit team data
        # In a real scenario, you'd extract team information from the data
        teams = [
            {'team_id': 1, 'name': 'New York Yankees', 'abbreviation': 'NYY', 'league': 'American', 'division': 'East'},
            {'team_id': 2, 'name': 'Toronto Blue Jays', 'abbreviation': 'TOR', 'league': 'American', 'division': 'East'},
            {'team_id': 3, 'name': 'Texas Rangers', 'abbreviation': 'TEX', 'league': 'American', 'division': 'West'},
            {'team_id': 4, 'name': 'Washington Nationals', 'abbreviation': 'WSH', 'league': 'National', 'division': 'East'},
            {'team_id': 5, 'name': 'Kansas City Royals', 'abbreviation': 'KC', 'league': 'American', 'division': 'Central'},
            {'team_id': 6, 'name': 'Cincinnati Reds', 'abbreviation': 'CIN', 'league': 'National', 'division': 'Central'},
        ]
        
        # Insert teams
        for team in teams:
            cur.execute(
                """INSERT INTO teams (team_id, name, abbreviation, league, division) 
                   VALUES (%s, %s, %s, %s, %s) 
                   ON CONFLICT (team_id) DO UPDATE 
                   SET name = EXCLUDED.name,
                       abbreviation = EXCLUDED.abbreviation,
                       league = EXCLUDED.league,
                       division = EXCLUDED.division""",
                (team['team_id'], team['name'], team['abbreviation'], team['league'], team['division'])
            )
            teams_added += 1
        
        # Assign players to teams (randomly for this example)
        import random
        team_ids = [team['team_id'] for team in teams]
        
        # Insert players
        for player in all_players:
            # Split name into first and last
            name_parts = player['name'].split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            # Assign to random team for this example
            team_id = random.choice(team_ids)
            
            cur.execute(
                """INSERT INTO players (player_id, team_id, first_name, last_name, position) 
                   VALUES (%s, %s, %s, %s, %s) 
                   ON CONFLICT (player_id) DO UPDATE 
                   SET team_id = EXCLUDED.team_id,
                       first_name = EXCLUDED.first_name,
                       last_name = EXCLUDED.last_name,
                       position = EXCLUDED.position""",
                (player['player_id'], team_id, first_name, last_name, player['position'])
            )
            players_added += 1
        
        conn.commit()
        logger.info(f"Added/updated {teams_added} teams and {players_added} players")
        
    except Exception as e:
        logger.error(f"Error processing teams and players: {e}")
        conn.rollback()
        raise

def process_games(conn, df):
    """Process and import game data"""
    logger.info("Processing games...")
    
    cur = conn.cursor()
    games_added = 0
    
    try:
        # Extract unique games
        games = df[['game_id', 'game_venue', 'game_date', 'game_datetime', 'temp']].drop_duplicates()
        
        # Insert games
        for _, game in games.iterrows():
            game_id = int(game['game_id'])
            venue = game['game_venue']
            date = game['game_date']
            
            # For this example, we'll assign random home/away teams
            # In a real scenario, you'd extract this from the data
            import random
            cur.execute("SELECT team_id FROM teams")
            team_ids = [row[0] for row in cur.fetchall()]
            
            if len(team_ids) < 2:
                logger.warning("Not enough teams to assign to games")
                continue
                
            home_team_id, away_team_id = random.sample(team_ids, 2)
            
            # Determine if venue is a dome (simplified)
            venue_is_dome = 'dome' in venue.lower() or 'field' in venue.lower()
            
            # Game time (simplified)
            game_time = "19:05"  # Default time
            if 'T' in game['game_datetime']:
                time_part = game['game_datetime'].split('T')[1][:5]
                if time_part:
                    game_time = time_part
            
            cur.execute(
                """INSERT INTO games (game_id, date, home_team_id, away_team_id, venue, venue_is_dome, game_time, status) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                   ON CONFLICT (game_id) DO UPDATE 
                   SET date = EXCLUDED.date,
                       home_team_id = EXCLUDED.home_team_id,
                       away_team_id = EXCLUDED.away_team_id,
                       venue = EXCLUDED.venue,
                       venue_is_dome = EXCLUDED.venue_is_dome,
                       game_time = EXCLUDED.game_time,
                       status = EXCLUDED.status""",
                (game_id, date, home_team_id, away_team_id, venue, venue_is_dome, game_time, "Scheduled")
            )
            games_added += 1
        
        conn.commit()
        logger.info(f"Added/updated {games_added} games")
        
    except Exception as e:
        logger.error(f"Error processing games: {e}")
        conn.rollback()
        raise

def process_player_stats(conn, df):
    """Process and import player statistics"""
    logger.info("Processing player statistics...")
    
    cur = conn.cursor()
    batting_stats_added = 0
    pitching_stats_added = 0
    
    try:
        # Process each row in the dataframe
        for _, row in df.iterrows():
            game_date = row['game_date']
            
            # Process batting stats
            player_id = int(row['batting_id'])
            
            # Check if we have meaningful batting stats
            if pd.notna(row.get('batting_avg')) and pd.notna(row.get('batting_atBats')):
                at_bats = int(row['batting_atBats']) if pd.notna(row['batting_atBats']) else 0
                hits = int(row['batting_hits']) if pd.notna(row['batting_hits']) else 0
                doubles = int(row['batting_doubles']) if pd.notna(row['batting_doubles']) else 0
                triples = int(row['batting_triples']) if pd.notna(row['batting_triples']) else 0
                home_runs = int(row['batting_homeRuns']) if pd.notna(row['batting_homeRuns']) else 0
                rbis = int(row['batting_rbi']) if pd.notna(row['batting_rbi']) else 0
                walks = int(row['batting_baseOnBalls']) if pd.notna(row['batting_baseOnBalls']) else 0
                strikeouts = int(row['batting_strikeOuts']) if pd.notna(row['batting_strikeOuts']) else 0
                batting_avg = float(row['batting_avg']) if pd.notna(row['batting_avg']) else 0.0
                on_base_pct = float(row['batting_obp']) if pd.notna(row['batting_obp']) else 0.0
                slugging_pct = float(row['batting_slg']) if pd.notna(row['batting_slg']) else 0.0
                ops = float(row['batting_ops']) if pd.notna(row['batting_ops']) else 0.0
                
                # Insert batting stats
                cur.execute(
                    """INSERT INTO player_batting_stats
                       (player_id, date, at_bats, hits, doubles, triples, 
                        home_runs, rbis, walks, strikeouts, batting_avg, 
                        on_base_pct, slugging_pct, ops, stat_period)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (player_id, date, stat_period) 
                       DO UPDATE
                       SET at_bats = EXCLUDED.at_bats,
                           hits = EXCLUDED.hits,
                           doubles = EXCLUDED.doubles,
                           triples = EXCLUDED.triples,
                           home_runs = EXCLUDED.home_runs,
                           rbis = EXCLUDED.rbis,
                           walks = EXCLUDED.walks,
                           strikeouts = EXCLUDED.strikeouts,
                           batting_avg = EXCLUDED.batting_avg,
                           on_base_pct = EXCLUDED.on_base_pct,
                           slugging_pct = EXCLUDED.slugging_pct,
                           ops = EXCLUDED.ops""",
                    (player_id, game_date, at_bats, hits, doubles, triples, home_runs, 
                     rbis, walks, strikeouts, batting_avg, on_base_pct, slugging_pct, ops, "Season")
                )
                batting_stats_added += 1
            
            # Process pitching stats
            pitcher_id = int(row['pitching_id'])
            
            # Check if we have meaningful pitching stats
            if pd.notna(row.get('pitching_era')) and pd.notna(row.get('pitching_inningsPitched')):
                innings = float(row['pitching_inningsPitched']) if pd.notna(row['pitching_inningsPitched']) else 0.0
                hits_allowed = int(row['pitching_hits']) if pd.notna(row['pitching_hits']) else 0
                runs = int(row['pitching_runs']) if pd.notna(row['pitching_runs']) else 0
                earned_runs = int(row['pitching_earnedRuns']) if pd.notna(row['pitching_earnedRuns']) else 0
                walks = int(row['pitching_baseOnBalls']) if pd.notna(row['pitching_baseOnBalls']) else 0
                strikeouts = int(row['pitching_strikeOuts']) if pd.notna(row['pitching_strikeOuts']) else 0
                era = float(row['pitching_era']) if pd.notna(row['pitching_era']) else 0.0
                whip = float(row['pitching_whip']) if pd.notna(row['pitching_whip']) else 0.0
                
                # Insert pitching stats
                cur.execute(
                    """INSERT INTO player_pitching_stats
                       (player_id, date, innings_pitched, hits_allowed, runs_allowed,
                        earned_runs, walks, strikeouts, era, whip, stat_period)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (player_id, date, stat_period) 
                       DO UPDATE
                       SET innings_pitched = EXCLUDED.innings_pitched,
                           hits_allowed = EXCLUDED.hits_allowed,
                           runs_allowed = EXCLUDED.runs_allowed,
                           earned_runs = EXCLUDED.earned_runs,
                           walks = EXCLUDED.walks,
                           strikeouts = EXCLUDED.strikeouts,
                           era = EXCLUDED.era,
                           whip = EXCLUDED.whip""",
                    (pitcher_id, game_date, innings, hits_allowed, runs, earned_runs, 
                     walks, strikeouts, era, whip, "Season")
                )
                pitching_stats_added += 1
        
        conn.commit()
        logger.info(f"Added/updated {batting_stats_added} batting stats and {pitching_stats_added} pitching stats")
        
    except Exception as e:
        logger.error(f"Error processing player statistics: {e}")
        conn.rollback()
        raise

def process_props_and_odds(conn, df):
    """Process and import prop betting odds"""
    logger.info("Processing prop betting odds...")
    
    cur = conn.cursor()
    odds_added = 0
    
    try:
        # Ensure prop types exist
        prop_types = [
            (1, 'Hits', 'Batting', 'Total hits by player'),
            (2, 'Home Runs', 'Batting', 'Total home runs by player'),
            (3, 'RBIs', 'Batting', 'Total runs batted in by player'),
            (4, 'Strikeouts', 'Pitching', 'Total strikeouts by pitcher'),
            (5, 'Earned Runs', 'Pitching', 'Total earned runs allowed by pitcher')
        ]
        
        for prop in prop_types:
            cur.execute(
                """INSERT INTO prop_types (prop_type_id, name, category, description) 
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (prop_type_id) DO NOTHING""",
                prop
            )
        
        # Ensure sportsbooks exist
        sportsbooks = [
            (1, 'DraftKings', 'draftkings'),
            (2, 'FanDuel', 'fanduel'),
            (3, 'BetMGM', 'betmgm'),
            (4, 'Caesars', 'caesars')
        ]
        
        for book in sportsbooks:
            cur.execute(
                """INSERT INTO sportsbooks (sportsbook_id, name, api_name) 
                   VALUES (%s, %s, %s)
                   ON CONFLICT (sportsbook_id) DO NOTHING""",
                book
            )
        
        # Generate odds from the dataframe
        # For this example, we'll create synthetic odds based on player stats
        for _, row in df.iterrows():
            game_id = int(row['game_id'])
            
            # Process batting props
            batter_id = int(row['batting_id'])
            
            # Hits prop
            if pd.notna(row.get('batting_avg')):
                # Generate a line based on batting average
                hits_line = 0.5 if row['batting_avg'] < 0.250 else 1.5
                
                # Generate odds
                import random
                over_odds = random.randint(-130, 110)
                under_odds = -110 if over_odds > 0 else 100
                
                # Insert hits prop odds
                cur.execute(
                    """INSERT INTO odds
                       (game_id, player_id, prop_type_id, sportsbook_id, line, over_odds, under_odds)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (game_id, player_id, prop_type_id, sportsbook_id) DO UPDATE
                       SET line = EXCLUDED.line,
                           over_odds = EXCLUDED.over_odds,
                           under_odds = EXCLUDED.under_odds,
                           timestamp = NOW()""",
                    (game_id, batter_id, 1, 1, hits_line, over_odds, under_odds)
                )
                odds_added += 1
            
            # Home runs prop
            if pd.notna(row.get('batting_homeRuns')):
                # Home runs are almost always 0.5
                hr_line = 0.5
                
                # Generate odds based on home run history
                import random
                over_odds = random.randint(-110, 200)
                under_odds = -130 if over_odds > 0 else -150
                
                # Insert home run prop odds
                cur.execute(
                    """INSERT INTO odds
                       (game_id, player_id, prop_type_id, sportsbook_id, line, over_odds, under_odds)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (game_id, player_id, prop_type_id, sportsbook_id) DO UPDATE
                       SET line = EXCLUDED.line,
                           over_odds = EXCLUDED.over_odds,
                           under_odds = EXCLUDED.under_odds,
                           timestamp = NOW()""",
                    (game_id, batter_id, 2, 2, hr_line, over_odds, under_odds)
                )
                odds_added += 1
            
            # Process pitching props
            pitcher_id = int(row['pitching_id'])
            
            # Strikeouts prop
            if pd.notna(row.get('pitching_strikeOuts')) and pd.notna(row.get('pitching_inningsPitched')):
                # Calculate K/9 and set a line
                k9 = row['pitching_strikeOuts'] * 9 / row['pitching_inningsPitched'] if row['pitching_inningsPitched'] > 0 else 0
                k_line = 3.5 if k9 < 7 else (5.5 if k9 < 9 else 7.5)
                
                # Generate odds
                import random
                over_odds = random.randint(-120, 110)
                under_odds = -110 if over_odds > 0 else -120
                
                # Insert strikeout prop odds
                cur.execute(
                    """INSERT INTO odds
                       (game_id, player_id, prop_type_id, sportsbook_id, line, over_odds, under_odds)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (game_id, player_id, prop_type_id, sportsbook_id) DO UPDATE
                       SET line = EXCLUDED.line,
                           over_odds = EXCLUDED.over_odds,
                           under_odds = EXCLUDED.under_odds,
                           timestamp = NOW()""",
                    (game_id, pitcher_id, 4, 3, k_line, over_odds, under_odds)
                )
                odds_added += 1
        
        conn.commit()
        logger.info(f"Added/updated {odds_added} prop betting odds")
        
    except Exception as e:
        logger.error(f"Error processing prop betting odds: {e}")
        conn.rollback()
        raise

def main():
    """Main function to run the data import"""
    config = load_config()
    
    # Default path to the CSV file
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'example_data.csv')
    
    # Allow command-line override of the path
    import argparse
    parser = argparse.ArgumentParser(description='Import MLB prop betting data from CSV')
    parser.add_argument('--csv', default=default_path, help='Path to the CSV file')
    parser.add_argument('--reset', action='store_true', help='Reset database tables before importing')
    args = parser.parse_args()
    
    # Import the data
    success = import_dataframe(config, args.csv, args.reset)
    
    if success:
        logger.info("Data import completed successfully")
    else:
        logger.error("Data import failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
