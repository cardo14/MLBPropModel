import requests
import time
import psycopg2
from config import logger
from db_utils import connect_to_db

def get_team_list(config, team_id=None):
    """Get list of teams from database"""
    conn = connect_to_db(config)
    if not conn:
        return []
        
    cur = conn.cursor()
    try:
        if team_id:
            cur.execute("SELECT team_id, name FROM teams WHERE team_id = %s", (team_id,))
        else:
            cur.execute("SELECT team_id, name FROM teams")
            
        teams = cur.fetchall()
        return teams
    except psycopg2.Error as e:
        logger.error(f"Error fetching teams: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def fetch_mlb_teams(config):
    """Fetch MLB teams from the MLB Stats API"""
    logger.info("Fetching MLB teams...")
    url = f"{config['api_base_url']}/teams"
    params = {"sportId": 1}  # MLB is sportId 1
    
    conn = None
    cur = None
    
    try:
        # Make API request
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
        data = response.json()
        
        # Validate response structure
        if "teams" not in data:
            logger.error("Error: Unexpected API response format - 'teams' key not found")
            return
        
        # Connect to database
        conn = connect_to_db(config)
        if not conn:
            return
            
        cur = conn.cursor()
        teams_added = 0
        
        # Process each team
        for team in data["teams"]:
            # Skip non-MLB teams or inactive teams
            if team.get("sport", {}).get("id") != 1 or not team.get("active", True):
                continue
                
            team_id = team.get("id")
            if not team_id:
                continue
                
            name = team.get("name", "")
            abbreviation = team.get("abbreviation", "")
            
            # Get league and division
            league = team.get("league", {}).get("name", "") if team.get("league") else ""
            division = team.get("division", {}).get("name", "") if team.get("division") else ""
            
            # Debug output
            logger.debug(f"Team: {name}, League: {league}, Division: {division}")
            
            # Insert into database
            try:
                cur.execute(
                    """INSERT INTO teams (team_id, name, abbreviation, league, division) 
                       VALUES (%s, %s, %s, %s, %s) 
                       ON CONFLICT (team_id) DO UPDATE 
                       SET name = EXCLUDED.name,
                           abbreviation = EXCLUDED.abbreviation,
                           league = EXCLUDED.league,
                           division = EXCLUDED.division""",
                    (team_id, name, abbreviation, league, division)
                )
                teams_added += 1
            except psycopg2.Error as e:
                logger.error(f"Error inserting team {name}: {e}")
                continue
        
        conn.commit()
        logger.info(f"Teams successfully imported! Added/updated {teams_added} teams.")
        return teams_added
        
    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request error: {req_err}")
    except psycopg2.Error as db_err:
        logger.error(f"Database error: {db_err}")
        if conn:
            conn.rollback()  # Rollback transaction on error
    except Exception as e:
        logger.error(f"Unexpected error in fetch_mlb_teams: {e}")
    finally:
        # Ensure resources are properly closed
        if cur:
            cur.close()
        if conn:
            conn.close()

def fetch_players(config, team_id=None, use_alternative=False):
    """Fetch player data for all teams or a specific team"""
    method = "alternative" if use_alternative else "standard"
    logger.info(f"Fetching player data using {method} method...")
    
    # Get teams to process
    teams = get_team_list(config, team_id)
    
    if not teams:
        logger.error("No teams found. Run fetch_mlb_teams first.")
        return 0
        
    players_added = 0
    
    # Connect to database
    conn = connect_to_db(config)
    if not conn:
        return 0
        
    cur = conn.cursor()
    
    try:
        # Process each team
        for team_row in teams:
            tid = team_row[0]
            team_name = team_row[1]
            logger.info(f"Fetching players for {team_name} (ID: {tid})...")
            
            if use_alternative:
                # Alternative method: team roster endpoint
                url = f"{config['api_base_url']}/teams/{tid}/roster/fullRoster"
                params = {}
            else:
                # Standard method: players search endpoint
                url = f"{config['api_base_url']}/sports/1/players"
                params = {"season": 2025, "gameType": "R"}
            
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Process data based on method
                if use_alternative:
                    if "roster" not in data:
                        logger.warning(f"No roster data found for team {team_name}")
                        continue
                        
                    players_list = data["roster"]
                    logger.info(f"Found {len(players_list)} players for {team_name}")
                    
                    for player in players_list:
                        if "person" not in player:
                            continue
                            
                        player_id = player["person"]["id"]
                        name_parts = player["person"]["fullName"].split()
                        
                        if len(name_parts) > 1:
                            first_name = name_parts[0]
                            last_name = " ".join(name_parts[1:])
                        else:
                            first_name = player["person"]["fullName"]
                            last_name = ""
                            
                        position = player.get("position", {}).get("abbreviation", "")
                        jersey_number = player.get("jerseyNumber", "")
                        bats = ""
                        throws = ""
                else:
                    if "people" not in data:
                        logger.warning("Error: Unexpected API response format - 'people' key not found")
                        continue
                        
                    # Get team-specific players
                    players_list = [p for p in data["people"] if p.get("currentTeam", {}).get("id") == tid]
                    logger.info(f"Found {len(players_list)} players for {team_name}")
                    
                    for player in players_list:
                        player_id = player.get("id")
                        if not player_id:
                            continue
                            
                        first_name = player.get("firstName", "")
                        last_name = player.get("lastName", "")
                        position = player.get("primaryPosition", {}).get("abbreviation", "")
                        jersey_number = player.get("primaryNumber", "")
                        bats = player.get("batSide", {}).get("code", "")
                        throws = player.get("pitchHand", {}).get("code", "")
                
                # Insert player data
                for i, player_data in enumerate(players_list):
                    try:
                        if use_alternative:
                            player = player_data
                            player_id = player["person"]["id"]
                            name_parts = player["person"]["fullName"].split()
                            
                            if len(name_parts) > 1:
                                first_name = name_parts[0]
                                last_name = " ".join(name_parts[1:])
                            else:
                                first_name = player["person"]["fullName"]
                                last_name = ""
                                
                            position = player.get("position", {}).get("abbreviation", "")
                            jersey_number = player.get("jerseyNumber", "")
                            bats = ""
                            throws = ""
                        else:
                            player = player_data
                            player_id = player.get("id")
                            if not player_id:
                                continue
                                
                            first_name = player.get("firstName", "")
                            last_name = player.get("lastName", "")
                            position = player.get("primaryPosition", {}).get("abbreviation", "")
                            jersey_number = player.get("primaryNumber", "")
                            bats = player.get("batSide", {}).get("code", "")
                            throws = player.get("pitchHand", {}).get("code", "")
                        
                        # Debug output (limit to first 5 players to avoid console spam)
                        if i < 5:
                            logger.debug(f"  Player: {first_name} {last_name}, Position: {position}")
                        
                        cur.execute(
                            """INSERT INTO players 
                               (player_id, team_id, first_name, last_name, position, 
                                jersey_number, bats, throws) 
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (player_id) DO UPDATE 
                               SET team_id = EXCLUDED.team_id,
                                   first_name = EXCLUDED.first_name,
                                   last_name = EXCLUDED.last_name,
                                   position = EXCLUDED.position,
                                   jersey_number = EXCLUDED.jersey_number,
                                   bats = EXCLUDED.bats,
                                   throws = EXCLUDED.throws""",
                            (player_id, tid, first_name, last_name, position, 
                             jersey_number, bats, throws)
                        )
                        players_added += 1
                    except psycopg2.Error as e:
                        logger.error(f"Error inserting player {first_name} {last_name}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing player data: {e}")
                        continue
                
            except requests.exceptions.RequestException as req_err:
                logger.error(f"API request error for team {tid}: {req_err}")
                continue
            
            # Sleep briefly to avoid rate limiting
            time.sleep(1)
        
        conn.commit()
        logger.info(f"Players successfully imported! Added/updated {players_added} players.")
        return players_added
        
    except Exception as e:
        logger.error(f"Unexpected error in fetch_players: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
