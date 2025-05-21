import requests
import time
import psycopg2
from utils.config import logger
from utils.db_utils import connect_to_db

def fetch_prop_betting_odds(config):
    """Fetch player prop odds from sportsbooks"""
    logger.info("Fetching prop betting odds...")
    
    # Check if API key is configured
    if not config['sportsbook_api_key'] or not config['sportsbook_api_url']:
        logger.warning("Sportsbook API key or URL not configured. Using mock data for testing.")
        print("NOTE: Using mock betting odds data for testing purposes.")
        use_mock_data = True
    else:
        use_mock_data = False
    
    conn = connect_to_db(config)
    if not conn:
        return 0
        
    cur = conn.cursor()
    odds_added = 0
    
    try:
        # Get upcoming games
        cur.execute("""
            SELECT game_id, date, home_team_id, away_team_id 
            FROM games 
            WHERE date >= CURRENT_DATE
            ORDER BY date ASC
            LIMIT 10
        """)
        upcoming_games = cur.fetchall()
        
        if not upcoming_games:
            logger.warning("No upcoming games found. Please run fetch_game_schedule first.")
            print("No upcoming games found. Please run fetch_game_schedule first.")
            return 0
        
        # Get prop types to fetch
        cur.execute("SELECT prop_type_id, name FROM prop_types")
        prop_types = {row[0]: row[1] for row in cur.fetchall()}
        
        # Get sportsbooks
        cur.execute("SELECT sportsbook_id, api_name FROM sportsbooks")
        sportsbooks = {row[1]: row[0] for row in cur.fetchall()}
        
        # For each game, fetch props
        for game in upcoming_games:
            game_id, game_date, home_team_id, away_team_id = game
            
            # Get players for both teams
            cur.execute("""
                SELECT p.player_id, p.first_name, p.last_name, p.position, t.name as team_name
                FROM players p
                JOIN teams t ON p.team_id = t.team_id
                WHERE p.team_id IN (%s, %s)
            """, (home_team_id, away_team_id))
            
            players = cur.fetchall()
            
            if use_mock_data:
                # Generate mock data for testing
                mock_data = generate_mock_odds_data(players, prop_types, sportsbooks)
                odds_data = mock_data
            else:
                # Here you would make API calls to the sportsbook API
                # This is a placeholder for the actual API integration
                headers = {
                    "X-API-KEY": config['sportsbook_api_key']
                }
                
                # Placeholder for API request - replace with actual endpoint
                api_url = f"{config['sportsbook_api_url']}/games/{game_id}/props"
                
                try:
                    # Actual API call would go here
                    # response = requests.get(api_url, headers=headers, timeout=15)
                    # response.raise_for_status()
                    # odds_data = response.json()
                    
                    # For now, use mock data
                    odds_data = generate_mock_odds_data(players, prop_types, sportsbooks)
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"API error fetching odds for game {game_id}: {e}")
                    continue
            
            # Process odds data
            for odds_entry in odds_data:
                player_id = odds_entry["player_id"]
                prop_type_id = odds_entry["prop_type_id"]
                sportsbook_id = odds_entry["sportsbook_id"]
                line = odds_entry["line"]
                over_odds = odds_entry["over_odds"]
                under_odds = odds_entry["under_odds"]
                
                # Insert odds into database
                cur.execute("""
                    INSERT INTO odds
                    (game_id, player_id, prop_type_id, sportsbook_id, line, over_odds, under_odds)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id, player_id, prop_type_id, sportsbook_id) DO UPDATE
                    SET line = EXCLUDED.line,
                        over_odds = EXCLUDED.over_odds,
                        under_odds = EXCLUDED.under_odds,
                        timestamp = NOW()
                """, (game_id, player_id, prop_type_id, sportsbook_id, line, over_odds, under_odds))
                
                odds_added += 1
            
            # Sleep briefly to avoid rate limiting
            time.sleep(0.5)
        
        conn.commit()
        logger.info(f"Betting odds imported! Added/updated {odds_added} odds entries.")
        return odds_added
        
    except Exception as e:
        logger.error(f"Error in fetch_prop_betting_odds: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def generate_mock_odds_data(players, prop_types, sportsbooks):
    """Generate mock odds data for testing purposes"""
    import random
    
    mock_data = []
    
    # For each player, generate odds for relevant props
    for player in players:
        player_id = player[0]
        position = player[3]
        
        # Determine which props are relevant based on position
        relevant_props = []
        if position in ["P", "SP", "RP"]:
            # Pitching props
            relevant_props = [prop_id for prop_id, name in prop_types.items() 
                             if "Strikeout" in name or "Earned Run" in name]
        else:
            # Batting props
            relevant_props = [prop_id for prop_id, name in prop_types.items() 
                             if "Hit" in name or "Home Run" in name or "RBI" in name]
        
        # Generate odds for each relevant prop from each sportsbook
        for prop_id in relevant_props:
            for sportsbook_name, sportsbook_id in sportsbooks.items():
                # Generate realistic lines based on prop type
                if "Strikeout" in prop_types[prop_id]:
                    line = round(random.uniform(3.5, 8.5) * 2) / 2
                elif "Hit" in prop_types[prop_id]:
                    line = round(random.uniform(0.5, 2.5) * 2) / 2
                elif "Home Run" in prop_types[prop_id]:
                    line = 0.5
                elif "RBI" in prop_types[prop_id]:
                    line = round(random.uniform(0.5, 2.5) * 2) / 2
                else:
                    line = round(random.uniform(0.5, 5.5) * 2) / 2
                
                # Generate realistic odds
                over_odds = random.randint(-120, 120)
                if over_odds > 0:
                    under_odds = -110
                else:
                    under_odds = 100
                
                mock_data.append({
                    "player_id": player_id,
                    "prop_type_id": prop_id,
                    "sportsbook_id": sportsbook_id,
                    "line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds
                })
    
    return mock_data

def generate_predictions(config):
    """Generate prop betting predictions based on available data"""
    logger.info("Generating predictions...")
    
    conn = connect_to_db(config)
    if not conn:
        return 0
        
    cur = conn.cursor()
    predictions_added = 0
    
    try:
        # 1. Get prop types to model
        cur.execute("SELECT prop_type_id, name FROM prop_types WHERE category IN ('Batting', 'Pitching')")
        prop_types = cur.fetchall()
        
        # 2. Get upcoming games with odds
        cur.execute("""
            SELECT DISTINCT o.game_id, g.date, o.player_id, p.first_name, p.last_name, 
                           o.prop_type_id, pt.name as prop_name, o.line
            FROM odds o
            JOIN games g ON o.game_id = g.game_id
            JOIN players p ON o.player_id = p.player_id
            JOIN prop_types pt ON o.prop_type_id = pt.prop_type_id
            WHERE g.date >= CURRENT_DATE
            ORDER BY g.date, p.last_name, pt.name
        """)
        
        odds_data = cur.fetchall()
        
        # 3. For each player/prop combo, generate a prediction
        for odds_row in odds_data:
            game_id, game_date, player_id, first_name, last_name, prop_type_id, prop_name, line = odds_row
            
            # Get player stats based on prop type
            if "Hit" in prop_name or "Home Run" in prop_name or "RBI" in prop_name:
                # Get batting stats
                cur.execute("""
                    SELECT batting_avg, on_base_pct, slugging_pct, ops, 
                           hits, home_runs, rbis, at_bats
                    FROM player_batting_stats
                    WHERE player_id = %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (player_id,))
                
                stats = cur.fetchone()
                
                if not stats:
                    # No stats available, use the line as prediction
                    predicted_value = line
                else:
                    # Simple prediction model based on stats
                    batting_avg, obp, slg, ops, hits, home_runs, rbis, at_bats = stats
                    
                    if "Hit" in prop_name:
                        # Predict hits based on batting average
                        predicted_value = round(batting_avg * 4 * 10) / 10  # Assuming ~4 at-bats per game
                    elif "Home Run" in prop_name:
                        # Predict home runs based on HR rate
                        hr_rate = home_runs / at_bats if at_bats > 0 else 0.05
                        predicted_value = round(hr_rate * 4 * 10) / 10  # Assuming ~4 at-bats per game
                    elif "RBI" in prop_name:
                        # Predict RBIs based on RBI rate
                        rbi_rate = rbis / at_bats if at_bats > 0 else 0.2
                        predicted_value = round(rbi_rate * 4 * 10) / 10  # Assuming ~4 at-bats per game
                    else:
                        predicted_value = line
            
            elif "Strikeout" in prop_name:
                # Get pitching stats
                cur.execute("""
                    SELECT strikeouts, innings_pitched
                    FROM player_pitching_stats
                    WHERE player_id = %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (player_id,))
                
                stats = cur.fetchone()
                
                if not stats:
                    # No stats available, use the line as prediction
                    predicted_value = line
                else:
                    # Simple prediction model based on stats
                    strikeouts, innings_pitched = stats
                    
                    # Predict strikeouts based on K/IP rate
                    k_per_ip = strikeouts / float(innings_pitched) if float(innings_pitched) > 0 else 1
                    predicted_value = round(k_per_ip * 6 * 10) / 10  # Assuming ~6 innings per start
            else:
                predicted_value = line
            
            # Calculate confidence (placeholder - real model would be more sophisticated)
            confidence = 0.7
            
            # Calculate EV based on predicted value vs line
            if predicted_value > line:
                # Over is predicted
                ev_over = 0.1  # Placeholder
                ev_under = -0.1
            else:
                # Under is predicted
                ev_over = -0.1
                ev_under = 0.1
            
            # Insert prediction
            cur.execute("""
                INSERT INTO predictions
                (game_id, player_id, prop_type_id, predicted_value, 
                 predicted_probability, confidence_level, model_id,
                 ev_over, ev_under, timestamp)
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (game_id, player_id, prop_type_id, predicted_value, 
                  0.5, confidence, 1, ev_over, ev_under))
            
            predictions_added += 1
        
        conn.commit()
        logger.info(f"Predictions generated! Added {predictions_added} predictions.")
        return predictions_added
        
    except Exception as e:
        logger.error(f"Error in generate_predictions: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def find_value_bets(config, min_edge=0.1):
    """Find value bets based on predictions vs odds"""
    logger.info(f"Finding value bets with edge > {min_edge}...")
    
    conn = connect_to_db(config)
    if not conn:
        return
        
    cur = conn.cursor()
    
    try:
        query = """
        SELECT 
            g.date,
            CONCAT(p.first_name, ' ', p.last_name) as player,
            t.name as team,
            pt.name as prop,
            sb.name as sportsbook,
            o.line,
            o.over_odds,
            o.under_odds,
            pred.predicted_value,
            pred.confidence_level,
            CASE 
                WHEN pred.predicted_value > o.line THEN 'Over' 
                ELSE 'Under' 
            END as recommendation,
            CASE 
                WHEN pred.predicted_value > o.line THEN pred.ev_over
                ELSE pred.ev_under
            END as expected_value
        FROM predictions pred
        JOIN odds o ON pred.game_id = o.game_id 
            AND pred.player_id = o.player_id 
            AND pred.prop_type_id = o.prop_type_id
        JOIN players p ON pred.player_id = p.player_id
        JOIN teams t ON p.team_id = t.team_id
        JOIN prop_types pt ON pred.prop_type_id = pt.prop_type_id
        JOIN sportsbooks sb ON o.sportsbook_id = sb.sportsbook_id
        JOIN games g ON pred.game_id = g.game_id
        WHERE g.date >= CURRENT_DATE
          AND ABS(pred.predicted_value - o.line) / CASE WHEN o.line = 0 THEN 1 ELSE o.line END > %s
          AND pred.timestamp > NOW() - INTERVAL '24 hours'
          AND o.timestamp > NOW() - INTERVAL '6 hours'
        ORDER BY expected_value DESC, g.date, player
        LIMIT 20
        """
        
        cur.execute(query, (min_edge,))
        value_bets = cur.fetchall()
        
        if not value_bets:
            logger.info("No value bets found matching criteria.")
            return
            
        print("\n===== VALUE BETS =====")
        print(f"{'DATE':<12} {'PLAYER':<30} {'PROP':<15} {'LINE':<8} {'PRED':<8} {'REC':<6} {'SPORTSBOOK':<15}")
        print("-" * 100)
        
        for bet in value_bets:
            date, player, team, prop, sportsbook, line, over_odds, under_odds, pred_value, confidence, rec, ev = bet
            print(f"{date:<12} {player:<30} {prop:<15} {line:<8.1f} {pred_value:<8.2f} {rec:<6} {sportsbook:<15}")
            
        print(f"\nFound {len(value_bets)} potential value bets.")
        
    except Exception as e:
        logger.error(f"Error in find_value_bets: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
