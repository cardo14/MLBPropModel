"""
Import Example Dataframe
Imports data from the Example_Dataframe_May_5 CSV file into the SQL database
"""

import os
import pandas as pd
import logging
from utils.config import logger
from utils.db_utils import connect_to_db

def import_dataframe_to_db(config):
    """Import the example dataframe into the SQL database"""
    logger.info("Importing example dataframe into database...")
    
    # Path to the example dataframe
    dataframe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'data', 'Example_Dataframe_May_5')
    
    try:
        # Read the CSV file
        logger.info(f"Reading dataframe from {dataframe_path}")
        
        # Define column names based on the data structure
        # Note: These column names are inferred from the data structure and may need adjustment
        columns = [
            # First section - pitcher stats
            'row_id', 'games_played', 'games_started', 'at_bats', 'plate_appearances', 'hits', 
            'doubles', 'triples', 'home_runs', 'strikeouts', 'walks', 'hit_by_pitch', 'singles', 
            'extra_base_hits', 'batting_avg', 'total_bases', 'on_base_pct', 'slugging_pct', 'ops',
            'stolen_bases', 'caught_stealing', 'stolen_base_pct', 'sac_flies', 'pitches_seen', 
            'era', 'innings_pitched', 'wins', 'losses', 'saves', 'blown_saves', 'holds', 'complete_games',
            'strikeouts_pitched', 'whip', 'batters_faced', 'plate_appearances_pitcher', 'games_pitched',
            'shutouts', 'complete_games_2', 'total_bases_allowed', 'batting_avg_against', 'wins_2',
            'losses_2', 'quality_starts', 'blown_saves_2', 'pitches_thrown', 'strike_pct', 'win_pct',
            'k_per_9', 'bb_per_9', 'hr_per_9', 'k_per_bb', 'h_per_9', 'k_pct', 'bb_pct', 'hr_pct',
            'babip', 'fip', 'war', 'pitcher_name', 'pitcher_id',
            
            # Second section - batter stats
            'games_played_batter', 'at_bats_batter', 'plate_appearances_batter', 'hits_batter',
            'doubles_batter', 'triples_batter', 'home_runs_batter', 'strikeouts_batter', 'walks_batter',
            'hit_by_pitch_batter', 'singles_batter', 'extra_base_hits_batter', 'batting_avg_batter',
            'total_bases_batter', 'on_base_pct_batter', 'slugging_pct_batter', 'ops_batter',
            'stolen_bases_batter', 'caught_stealing_batter', 'stolen_base_pct_batter', 'sac_flies_batter',
            'pitches_seen_batter', 'total_pitches_batter', 'total_swings_batter', 'zone_swings_batter',
            'zone_pitches_batter', 'first_pitch_swings_batter', 'chase_pitches_batter', 'contact_pct_batter',
            'zone_contact_pct_batter', 'chase_contact_pct_batter', 'swing_pct_batter', 'batter_name',
            'batter_id', 'line', 'over_odds', 'under_odds', 'over_probability', 'under_probability',
            'vig', 'venue_id', 'venue_name', 'game_date', 'game_time', 'temperature'
        ]
        
        # Read the CSV file with headers
        df = pd.read_csv(dataframe_path, header=0)
        
        # Connect to the database
        conn = connect_to_db(config)
        if not conn:
            logger.error("Failed to connect to database")
            return False
        
        cur = conn.cursor()
        
        # Import data into the database tables
        records_added = {
            'teams': 0,
            'players': 0,
            'games': 0,
            'prop_types': 0,
            'sportsbooks': 0,
            'odds': 0
        }
        
        # 1. First, ensure we have the prop types in the database
        prop_types = [
            (1, 'Hits', 'Batting', 'Total hits by player'),
            (2, 'Home Runs', 'Batting', 'Total home runs by player'),
            (3, 'RBIs', 'Batting', 'Total runs batted in by player'),
            (4, 'Strikeouts', 'Pitching', 'Total strikeouts by pitcher'),
            (5, 'Earned Runs', 'Pitching', 'Total earned runs allowed by pitcher')
        ]
        
        for prop in prop_types:
            try:
                cur.execute(
                    """INSERT INTO prop_types (prop_type_id, name, category, description) 
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (prop_type_id) DO NOTHING""",
                    prop
                )
                records_added['prop_types'] += 1
            except Exception as e:
                logger.error(f"Error inserting prop type {prop[1]}: {e}")
        
        # 2. Ensure we have a default sportsbook
        sportsbooks = [
            (1, 'Example Sportsbook', 'example')
        ]
        
        for book in sportsbooks:
            try:
                cur.execute(
                    """INSERT INTO sportsbooks (sportsbook_id, name, api_name) 
                       VALUES (%s, %s, %s)
                       ON CONFLICT (sportsbook_id) DO NOTHING""",
                    book
                )
                records_added['sportsbooks'] += 1
            except Exception as e:
                logger.error(f"Error inserting sportsbook {book[1]}: {e}")
        
        # 3. Process each row in the dataframe
        venues_processed = set()
        
        for _, row in df.iterrows():
            # 3.1 Process venue/game data
            venue_id = int(row['game_id'])
            venue_name = row['game_venue']
            game_date = row['game_date']
            game_time = row['game_datetime']
            
            if venue_id not in venues_processed:
                # Create a game entry
                try:
                    # First, ensure we have teams for home and away
                    # For simplicity, we'll create placeholder teams if they don't exist
                    home_team_id = 1  # Placeholder
                    away_team_id = 2  # Placeholder
                    
                    cur.execute(
                        """INSERT INTO teams (team_id, name, abbreviation) 
                           VALUES (%s, %s, %s)
                           ON CONFLICT (team_id) DO NOTHING""",
                        (home_team_id, "Home Team", "HOME")
                    )
                    
                    cur.execute(
                        """INSERT INTO teams (team_id, name, abbreviation) 
                           VALUES (%s, %s, %s)
                           ON CONFLICT (team_id) DO NOTHING""",
                        (away_team_id, "Away Team", "AWAY")
                    )
                    
                    records_added['teams'] += 2
                    
                    # Create game entry
                    cur.execute(
                        """INSERT INTO games 
                           (game_id, date, home_team_id, away_team_id, venue, game_time, status) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (game_id) DO NOTHING""",
                        (venue_id, game_date, home_team_id, away_team_id, venue_name, 
                         game_time.split('T')[1].split('Z')[0], "Scheduled")
                    )
                    
                    records_added['games'] += 1
                    venues_processed.add(venue_id)
                except Exception as e:
                    logger.error(f"Error processing venue/game {venue_id}: {e}")
            
            # 3.2 Process pitcher data
            pitcher_id = int(row['pitching_id'])
            pitcher_name = row['pitching_name']
            
            try:
                # Split name into first and last
                name_parts = pitcher_name.split(' ', 1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                
                # Insert pitcher
                cur.execute(
                    """INSERT INTO players 
                       (player_id, team_id, first_name, last_name, position) 
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (player_id) DO NOTHING""",
                    (pitcher_id, 1, first_name, last_name, "P")  # Assuming pitcher is on home team
                )
                
                records_added['players'] += 1
                
                # Insert pitcher stats
                cur.execute(
                    """INSERT INTO player_pitching_stats 
                       (player_id, date, innings_pitched, hits_allowed, runs_allowed,
                        earned_runs, walks, strikeouts, era, whip, stat_period)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (player_id, date, stat_period) DO NOTHING""",
                    (pitcher_id, game_date, float(row['pitching_inningsPitched']), int(row['pitching_hits']), 
                     int(row['pitching_runs']), int(row['pitching_earnedRuns']),
                     int(row['pitching_baseOnBalls']), int(row['pitching_strikeOuts']), 
                     float(row['pitching_era']), float(row['pitching_whip']), "Season")
                )
            except Exception as e:
                logger.error(f"Error processing pitcher {pitcher_id}: {e}")
            
            # 3.3 Process batter data
            batter_id = int(row['batting_id'])
            batter_name = row['batting_name']
            
            try:
                # Split name into first and last
                name_parts = batter_name.split(' ', 1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                
                # Insert batter
                cur.execute(
                    """INSERT INTO players 
                       (player_id, team_id, first_name, last_name, position) 
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (player_id) DO NOTHING""",
                    (batter_id, 2, first_name, last_name, "B")  # Assuming batter is on away team
                )
                
                records_added['players'] += 1
                
                # Insert batter stats
                cur.execute(
                    """INSERT INTO player_batting_stats 
                       (player_id, date, at_bats, hits, doubles, triples, 
                        home_runs, rbis, walks, strikeouts, batting_avg, 
                        on_base_pct, slugging_pct, ops, stat_period)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (player_id, date, stat_period) DO NOTHING""",
                    (batter_id, game_date, int(row['batting_atBats']), int(row['batting_hits']), 
                     int(row['batting_doubles']), int(row['batting_triples']), int(row['batting_homeRuns']),
                     int(row['batting_rbi']), int(row['batting_baseOnBalls']), int(row['batting_strikeOuts']), 
                     float(row['batting_avg']), float(row['batting_obp']), 
                     float(row['batting_slg']), float(row['batting_ops']), "Season")
                )
            except Exception as e:
                logger.error(f"Error processing batter {batter_id}: {e}")
            
            # 3.4 Process odds data
            try:
                # Determine prop type based on data - default to hits
                prop_type_id = 1
                
                # Convert odds values - handle potential string values
                try:
                    line = float(row['batting_line'])
                    over_odds = int(row['batting_over_odds'])
                    under_odds = int(row['batting_under_odds'])
                except (ValueError, KeyError):
                    # If we can't parse the odds values, use default values
                    line = 0.5
                    over_odds = -110
                    under_odds = -110
                
                # Insert odds
                cur.execute(
                    """INSERT INTO odds
                       (game_id, player_id, prop_type_id, sportsbook_id, line, over_odds, under_odds)
                       VALUES
                       (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (game_id, player_id, prop_type_id, sportsbook_id) DO NOTHING""",
                    (venue_id, batter_id, prop_type_id, 1, line, over_odds, under_odds)
                )
                
                records_added['odds'] += 1
            except Exception as e:
                logger.error(f"Error processing odds for player {batter_id}: {e}")
        
        # Commit all changes
        conn.commit()
        
        # Log summary
        logger.info("Import completed successfully!")
        for table, count in records_added.items():
            logger.info(f"Added {count} records to {table} table")
        
        return True
        
    except Exception as e:
        logger.error(f"Error importing dataframe: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    from utils.config import load_config
    config = load_config()
    import_dataframe_to_db(config)
