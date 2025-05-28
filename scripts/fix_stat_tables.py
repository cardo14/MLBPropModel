import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fix_stat_tables')

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mlb_prop_betting',
            user='postgres',
            password='ophadke'
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def fix_stat_tables():
    conn = connect_to_db()
    if not conn:
        return False
    cur = conn.cursor()
    try:
        # First, remove redundant constraints
        cur.execute("""
            ALTER TABLE player_batting_stats 
            DROP CONSTRAINT IF EXISTS player_batting_stats_player_id_season_key,
            DROP CONSTRAINT IF EXISTS player_batting_stats_player_id_season_stat_period_key;
            
            ALTER TABLE player_pitching_stats 
            DROP CONSTRAINT IF EXISTS player_pitching_stats_player_id_season_key,
            DROP CONSTRAINT IF EXISTS player_pitching_stats_player_id_season_stat_period_key;
        """)
        
        # Remove redundant name columns
        cur.execute("""
            ALTER TABLE player_batting_stats 
            DROP COLUMN IF EXISTS first_name,
            DROP COLUMN IF EXISTS last_name;
            
            ALTER TABLE player_pitching_stats 
            DROP COLUMN IF EXISTS first_name,
            DROP COLUMN IF EXISTS last_name;
        """)
        
        # Update any null dates to current date
        cur.execute("""
            UPDATE player_batting_stats 
            SET date = CURRENT_DATE 
            WHERE date IS NULL;
            
            UPDATE player_pitching_stats 
            SET date = CURRENT_DATE 
            WHERE date IS NULL;
        """)
        
        # Now make date columns NOT NULL
        cur.execute("""
            ALTER TABLE player_batting_stats 
            ALTER COLUMN date SET NOT NULL;
            
            ALTER TABLE player_pitching_stats 
            ALTER COLUMN date SET NOT NULL;
        """)
        
        # Add proper unique constraints
        cur.execute("""
            ALTER TABLE player_batting_stats 
            ADD CONSTRAINT player_batting_stats_unique_key 
            UNIQUE (player_id, date, stat_period);
            
            ALTER TABLE player_pitching_stats 
            ADD CONSTRAINT player_pitching_stats_unique_key 
            UNIQUE (player_id, date, stat_period);
        """)
        
        conn.commit()
        logger.info("Successfully fixed stat tables")
        return True
    except Exception as e:
        logger.error(f"Error fixing stat tables: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    fix_stat_tables() 