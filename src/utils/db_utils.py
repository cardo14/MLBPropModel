import psycopg2
import logging

# Import logger from the local config module
from utils.config import logger

def connect_to_db(config):
    """Connect to the PostgreSQL database server"""
    conn = None
    try:
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(
            host=config['db_host'],
            database=config['db_name'],
            user=config['db_user'],
            password=config['db_password']
        )
        return conn
    except Exception as error:
        logger.error(f"Error connecting to database: {error}")
        return None

def reset_tables(config):
    """Drop and recreate tables - USE WITH CAUTION"""
    conn = connect_to_db(config)
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Drop tables if they exist
        cur.execute("DROP TABLE IF EXISTS players CASCADE")
        cur.execute("DROP TABLE IF EXISTS teams CASCADE")
        cur.execute("DROP TABLE IF EXISTS player_batting_stats CASCADE")
        cur.execute("DROP TABLE IF EXISTS player_pitching_stats CASCADE")
        cur.execute("DROP TABLE IF EXISTS games CASCADE")
        cur.execute("DROP TABLE IF EXISTS prop_types CASCADE")
        cur.execute("DROP TABLE IF EXISTS sportsbooks CASCADE")
        cur.execute("DROP TABLE IF EXISTS odds CASCADE")
        cur.execute("DROP TABLE IF EXISTS predictions CASCADE")
        
        # Create teams table
        cur.execute("""
            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                abbreviation VARCHAR(50),
                league VARCHAR(50),
                division VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create players table
        cur.execute("""
            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY,
                team_id INTEGER REFERENCES teams(team_id),
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                position VARCHAR(50),
                jersey_number VARCHAR(20),
                bats VARCHAR(20),
                throws VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create player_batting_stats table
        cur.execute("""
            CREATE TABLE player_batting_stats (
                id SERIAL PRIMARY KEY,
                player_id INTEGER REFERENCES players(player_id),
                date DATE,
                at_bats INTEGER,
                hits INTEGER,
                doubles INTEGER,
                triples INTEGER,
                home_runs INTEGER,
                rbis INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                batting_avg FLOAT,
                on_base_pct FLOAT,
                slugging_pct FLOAT,
                ops FLOAT,
                stat_period VARCHAR(20),
                UNIQUE(player_id, date, stat_period)
            )
        """)
        
        # Create player_pitching_stats table
        cur.execute("""
            CREATE TABLE player_pitching_stats (
                id SERIAL PRIMARY KEY,
                player_id INTEGER REFERENCES players(player_id),
                date DATE,
                innings_pitched FLOAT,
                hits_allowed INTEGER,
                runs_allowed INTEGER,
                earned_runs INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                era FLOAT,
                whip FLOAT,
                stat_period VARCHAR(20),
                UNIQUE(player_id, date, stat_period)
            )
        """)
        
        # Create games table
        cur.execute("""
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY,
                date DATE,
                home_team_id INTEGER REFERENCES teams(team_id),
                away_team_id INTEGER REFERENCES teams(team_id),
                venue VARCHAR(100),
                venue_is_dome BOOLEAN,
                game_time VARCHAR(10),
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create prop_types table
        cur.execute("""
            CREATE TABLE prop_types (
                prop_type_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create sportsbooks table
        cur.execute("""
            CREATE TABLE sportsbooks (
                sportsbook_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                api_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create odds table
        cur.execute("""
            CREATE TABLE odds (
                id SERIAL PRIMARY KEY,
                game_id INTEGER REFERENCES games(game_id),
                player_id INTEGER REFERENCES players(player_id),
                prop_type_id INTEGER REFERENCES prop_types(prop_type_id),
                sportsbook_id INTEGER REFERENCES sportsbooks(sportsbook_id),
                line FLOAT,
                over_odds INTEGER,
                under_odds INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_id, player_id, prop_type_id, sportsbook_id)
            )
        """)
        
        # Create predictions table
        cur.execute("""
            CREATE TABLE predictions (
                id SERIAL PRIMARY KEY,
                game_id INTEGER REFERENCES games(game_id),
                player_id INTEGER REFERENCES players(player_id),
                prop_type_id INTEGER REFERENCES prop_types(prop_type_id),
                predicted_value FLOAT,
                predicted_probability FLOAT,
                confidence_level FLOAT,
                model_id INTEGER,
                ev_over FLOAT,
                ev_under FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_id, player_id, prop_type_id, model_id)
            )
        """)
        
        # Insert default prop types
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
                   VALUES (%s, %s, %s, %s)""",
                prop
            )
        
        # Insert default sportsbooks
        sportsbooks = [
            (1, 'DraftKings', 'draftkings'),
            (2, 'FanDuel', 'fanduel'),
            (3, 'BetMGM', 'betmgm'),
            (4, 'Caesars', 'caesars')
        ]
        
        for book in sportsbooks:
            cur.execute(
                """INSERT INTO sportsbooks (sportsbook_id, name, api_name) 
                   VALUES (%s, %s, %s)""",
                book
            )
        
        conn.commit()
        logger.info("Tables reset successfully!")
        return True
    except Exception as e:
        logger.error(f"Error resetting tables: {e}")
        if conn:
            conn.close()
        return False
    finally:
        if conn:
            conn.close()
