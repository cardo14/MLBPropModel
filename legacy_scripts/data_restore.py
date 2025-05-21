#!/usr/bin/env python3
"""
MLB Prop Data Restoration Script
--------------------------------
This script restores the MLB prop betting database with teams, players, and stats data.
It implements batching and error handling to avoid timeouts and API rate limiting issues.
"""

import time
import sys
import random
from config import logger, load_config
from db_utils import reset_tables, connect_to_db
from team_player_data import fetch_mlb_teams, fetch_players
from stats_games import fetch_game_schedule

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

def fetch_player_stats_batched(config, batch_size=50, delay_between_batches=5):
    """Fetch player statistics in smaller batches to avoid timeouts"""
    logger.info(f"Fetching player statistics in batches of {batch_size}...")
    
    conn = connect_to_db(config)
    if not conn:
        return 0
        
    cur = conn.cursor()
    
    try:
        # Get all players from database
        cur.execute("SELECT player_id, first_name, last_name FROM players")
        all_players = cur.fetchall()
        
        # Shuffle players to distribute the load more evenly
        random.shuffle(all_players)
        
        total_players = len(all_players)
        logger.info(f"Found {total_players} players to process")
        
        stats_added = 0
        batches = [all_players[i:i + batch_size] for i in range(0, total_players, batch_size)]
        
        for batch_num, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_num+1}/{len(batches)} ({len(batch)} players)")
            
            for player in batch:
                player_id = player[0]
                player_name = f"{player[1]} {player[2]}"
                
                # MLB Stats API endpoint for player stats
                url = f"{config['api_base_url']}/people/{player_id}/stats"
                params = {
                    "stats": "season",
                    "season": "2025",  # Current season
                    "group": "hitting,pitching",
                    "gameType": "R"    # Regular season
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code != 200:
                        logger.warning(f"API returned status {response.status_code} for player {player_name}")
                        continue
                        
                    data = response.json()
                    
                    if "stats" not in data:
                        continue
                        
                    # Process hitting stats
                    for stat_group in data["stats"]:
                        if "group" not in stat_group or "splits" not in stat_group:
                            continue
                            
                        group_name = stat_group["group"]["displayName"]
                        
                        if not stat_group["splits"]:
                            continue
                            
                        stats = stat_group["splits"][0]["stat"]
                        
                        if group_name == "hitting":
                            # Insert batting stats
                            at_bats = stats.get("atBats", 0)
                            hits = stats.get("hits", 0)
                            doubles = stats.get("doubles", 0)
                            triples = stats.get("triples", 0)
                            home_runs = stats.get("homeRuns", 0)
                            rbis = stats.get("rbi", 0)
                            walks = stats.get("baseOnBalls", 0)
                            strikeouts = stats.get("strikeOuts", 0)
                            batting_avg = stats.get("avg", 0)
                            obp = stats.get("obp", 0)
                            slg = stats.get("slg", 0)
                            ops = stats.get("ops", 0)
                            
                            # Insert to player_batting_stats table
                            cur.execute("""
                                INSERT INTO player_batting_stats
                                (player_id, date, at_bats, hits, doubles, triples, 
                                 home_runs, rbis, walks, strikeouts, batting_avg, 
                                 on_base_pct, slugging_pct, ops, stat_period)
                                VALUES
                                (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                                    ops = EXCLUDED.ops
                            """, (player_id, at_bats, hits, doubles, triples, home_runs, 
                                  rbis, walks, strikeouts, batting_avg, obp, slg, ops, "Season"))
                            
                            stats_added += 1
                            
                        elif group_name == "pitching":
                            # Insert pitching stats
                            innings = stats.get("inningsPitched", 0)
                            hits_allowed = stats.get("hits", 0)
                            runs = stats.get("runs", 0)
                            earned_runs = stats.get("earnedRuns", 0)
                            walks = stats.get("baseOnBalls", 0)
                            strikeouts = stats.get("strikeOuts", 0)
                            era = stats.get("era", 0)
                            whip = stats.get("whip", 0)
                            
                            # Insert to player_pitching_stats table
                            cur.execute("""
                                INSERT INTO player_pitching_stats
                                (player_id, date, innings_pitched, hits_allowed, runs_allowed,
                                 earned_runs, walks, strikeouts, era, whip, stat_period)
                                VALUES
                                (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (player_id, date, stat_period) 
                                DO UPDATE
                                SET innings_pitched = EXCLUDED.innings_pitched,
                                    hits_allowed = EXCLUDED.hits_allowed,
                                    runs_allowed = EXCLUDED.runs_allowed,
                                    earned_runs = EXCLUDED.earned_runs,
                                    walks = EXCLUDED.walks,
                                    strikeouts = EXCLUDED.strikeouts,
                                    era = EXCLUDED.era,
                                    whip = EXCLUDED.whip
                            """, (player_id, innings, hits_allowed, runs, earned_runs, 
                                  walks, strikeouts, era, whip, "Season"))
                            
                            stats_added += 1
                    
                except Exception as e:
                    logger.error(f"Error processing stats for player {player_name}: {e}")
                    continue
                    
                # Brief delay between player requests to avoid rate limiting
                time.sleep(0.2)
            
            # Commit after each batch
            conn.commit()
            logger.info(f"Completed batch {batch_num+1}/{len(batches)}, stats added so far: {stats_added}")
            
            # Sleep between batches to avoid overwhelming the API
            if batch_num < len(batches) - 1:
                logger.info(f"Sleeping for {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)
        
        logger.info(f"Player stats imported! Added/updated {stats_added} stat records.")
        return stats_added
        
    except Exception as e:
        logger.error(f"Error in fetch_player_stats_batched: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def restore_data(config):
    """Restore all necessary data to the database"""
    try:
        # Step 1: Reset tables
        logger.info("Step 1: Resetting database tables...")
        if not reset_tables(config):
            logger.error("Failed to reset tables. Aborting restoration.")
            return False
        
        # Step 2: Fetch teams
        logger.info("Step 2: Fetching MLB teams...")
        teams_added = fetch_mlb_teams(config)
        if not teams_added:
            logger.error("Failed to fetch teams. Aborting restoration.")
            return False
        logger.info(f"Successfully added {teams_added} teams.")
        
        # Step 3: Fetch players
        logger.info("Step 3: Fetching player data...")
        players_added = fetch_players(config)
        if not players_added:
            logger.warning("Standard player fetch method failed. Trying alternative method...")
            players_added = fetch_players(config, use_alternative=True)
            if not players_added:
                logger.error("Failed to fetch players with both methods. Aborting restoration.")
                return False
        logger.info(f"Successfully added {players_added} players.")
        
        # Step 4: Fetch player statistics with batching
        logger.info("Step 4: Fetching player statistics...")
        stats_added = fetch_player_stats_batched(config, batch_size=20, delay_between_batches=3)
        if stats_added:
            logger.info(f"Successfully added {stats_added} player statistic records.")
        else:
            logger.warning("No player statistics were added.")
        
        # Step 5: Fetch game schedule
        logger.info("Step 5: Fetching game schedule...")
        games_added = fetch_game_schedule(config, days_forward=14, days_back=7)
        if games_added:
            logger.info(f"Successfully added {games_added} game records.")
        else:
            logger.warning("No games were added. This may be normal during off-season.")
        
        logger.info("Data restoration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error during data restoration: {e}")
        return False

def verify_data_integrity(config):
    """Verify that data was properly restored by checking counts"""
    logger.info("Verifying data integrity...")
    conn = connect_to_db(config)
    if not conn:
        return False
    
    cur = conn.cursor()
    try:
        # Check team count
        cur.execute("SELECT COUNT(*) FROM teams")
        team_count = cur.fetchone()[0]
        
        # Check player count
        cur.execute("SELECT COUNT(*) FROM players")
        player_count = cur.fetchone()[0]
        
        # Check stats count
        cur.execute("SELECT COUNT(*) FROM player_batting_stats")
        batting_stats_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM player_pitching_stats")
        pitching_stats_count = cur.fetchone()[0]
        
        # Check game count
        cur.execute("SELECT COUNT(*) FROM games")
        game_count = cur.fetchone()[0]
        
        logger.info(f"Data verification results:")
        logger.info(f"- Teams: {team_count}")
        logger.info(f"- Players: {player_count}")
        logger.info(f"- Batting stats: {batting_stats_count}")
        logger.info(f"- Pitching stats: {pitching_stats_count}")
        logger.info(f"- Games: {game_count}")
        
        print(f"\nData verification results:")
        print(f"- Teams: {team_count}")
        print(f"- Players: {player_count}")
        print(f"- Batting stats: {batting_stats_count}")
        print(f"- Pitching stats: {pitching_stats_count}")
        print(f"- Games: {game_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying data integrity: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to run the data restoration process"""
    logger.info("Starting MLB Prop Betting data restoration")
    config = load_config()
    
    # Step 1: Verify database connection
    if not verify_db_connection(config):
        logger.error("Database connection verification failed. Please check your configuration.")
        sys.exit(1)
    
    # Step 2: Restore data
    if not restore_data(config):
        logger.error("Data restoration failed. Please check the logs for details.")
        sys.exit(1)
    
    # Step 3: Verify data integrity
    verify_data_integrity(config)
    
    logger.info("Data restoration process completed.")
    print("\nData restoration process completed successfully!")
    print("You can now run the MLB Prop Betting application.")

if __name__ == "__main__":
    import requests  # Import here to avoid circular imports
    main()
