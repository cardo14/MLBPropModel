#!/usr/bin/env python3
"""
MLB Prop Betting System
Main application entry point with menu-driven interface
"""

import os
import sys
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from utils.config import load_config, logger
from utils.db_utils import connect_to_db, reset_tables
from models.betting import generate_predictions, find_value_bets
from data.import_dataframe import import_dataframe_to_db

def display_menu():
    """Display the main menu options"""
    print("\n" + "=" * 50)
    print("MLB PROP BETTING SYSTEM")
    print("=" * 50)
    print("1. Data Management")
    print("2. Analysis & Predictions")
    print("3. Database Management")
    print("0. Exit")
    print("=" * 50)
    return input("Enter your choice: ")

def display_data_menu():
    """Display the data management menu options"""
    print("\n" + "=" * 50)
    print("DATA MANAGEMENT")
    print("=" * 50)
    print("1. View player statistics")
    print("2. View game schedule")
    print("3. View prop betting odds")
    print("4. Import data from CSV")
    print("5. Import example dataframe")
    print("0. Back to main menu")
    print("=" * 50)
    return input("Enter your choice: ")

def display_analysis_menu():
    """Display the analysis menu options"""
    print("\n" + "=" * 50)
    print("ANALYSIS & PREDICTIONS")
    print("=" * 50)
    print("1. Generate predictions")
    print("2. Find value bets")
    print("3. Export predictions to CSV")
    print("0. Back to main menu")
    print("=" * 50)
    return input("Enter your choice: ")

def display_db_menu():
    """Display the database management menu options"""
    print("\n" + "=" * 50)
    print("DATABASE MANAGEMENT")
    print("=" * 50)
    print("1. View database status")
    print("2. Reset database tables (CAUTION)")
    print("0. Back to main menu")
    print("=" * 50)
    return input("Enter your choice: ")

def view_player_stats(config):
    """View player statistics from the database"""
    conn = connect_to_db(config)
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # Get player list
        cur.execute("""
            SELECT p.player_id, p.first_name, p.last_name, p.position, t.name as team_name
            FROM players p
            JOIN teams t ON p.team_id = t.team_id
            ORDER BY p.last_name, p.first_name
        """)
        
        players = cur.fetchall()
        
        if not players:
            print("No players found in the database.")
            return
        
        print("\nPLAYERS:")
        for i, player in enumerate(players[:20], 1):
            print(f"{i}. {player[1]} {player[2]} ({player[3]}) - {player[4]}")
        
        if len(players) > 20:
            print(f"...and {len(players) - 20} more players")
        
        player_id = input("\nEnter player ID to view stats (or 0 to cancel): ")
        if player_id == "0":
            return
        
        try:
            player_id = int(player_id)
        except ValueError:
            print("Invalid input. Please enter a numeric ID.")
            return
        
        # Get player info
        cur.execute("""
            SELECT p.first_name, p.last_name, p.position, t.name as team_name
            FROM players p
            JOIN teams t ON p.team_id = t.team_id
            WHERE p.player_id = %s
        """, (player_id,))
        
        player_info = cur.fetchone()
        
        if not player_info:
            print(f"No player found with ID {player_id}")
            return
        
        print(f"\nSTATISTICS FOR {player_info[0]} {player_info[1]} ({player_info[2]}) - {player_info[3]}")
        
        # Check for batting stats
        cur.execute("""
            SELECT date, at_bats, hits, doubles, triples, home_runs, rbis, walks, strikeouts, 
                   batting_avg, on_base_pct, slugging_pct, ops, stat_period
            FROM player_batting_stats
            WHERE player_id = %s
            ORDER BY date DESC
            LIMIT 5
        """, (player_id,))
        
        batting_stats = cur.fetchall()
        
        if batting_stats:
            print("\nBATTING STATISTICS:")
            print(f"{'Date':<12} {'AB':<4} {'H':<4} {'2B':<4} {'3B':<4} {'HR':<4} {'RBI':<4} {'BB':<4} {'SO':<4} {'AVG':<6} {'OBP':<6} {'SLG':<6} {'OPS':<6} {'Period':<10}")
            print("-" * 90)
            
            for stat in batting_stats:
                print(f"{stat[0]!s:<12} {stat[1]:<4} {stat[2]:<4} {stat[3]:<4} {stat[4]:<4} {stat[5]:<4} {stat[6]:<4} {stat[7]:<4} {stat[8]:<4} {stat[9]:.3f} {stat[10]:.3f} {stat[11]:.3f} {stat[12]:.3f} {stat[13]:<10}")
        
        # Check for pitching stats
        cur.execute("""
            SELECT date, innings_pitched, hits_allowed, runs_allowed, earned_runs, walks, strikeouts, 
                   era, whip, stat_period
            FROM player_pitching_stats
            WHERE player_id = %s
            ORDER BY date DESC
            LIMIT 5
        """, (player_id,))
        
        pitching_stats = cur.fetchall()
        
        if pitching_stats:
            print("\nPITCHING STATISTICS:")
            print(f"{'Date':<12} {'IP':<5} {'H':<4} {'R':<4} {'ER':<4} {'BB':<4} {'SO':<4} {'ERA':<6} {'WHIP':<6} {'Period':<10}")
            print("-" * 70)
            
            for stat in pitching_stats:
                print(f"{stat[0]!s:<12} {stat[1]:<5.1f} {stat[2]:<4} {stat[3]:<4} {stat[4]:<4} {stat[5]:<4} {stat[6]:<4} {stat[7]:<6.2f} {stat[8]:<6.2f} {stat[9]:<10}")
        
        if not batting_stats and not pitching_stats:
            print("No statistics found for this player.")
        
    except Exception as e:
        logger.error(f"Error viewing player stats: {e}")
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

def view_game_schedule(config):
    """View game schedule from the database"""
    conn = connect_to_db(config)
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # Get upcoming games
        cur.execute("""
            SELECT g.game_id, g.date, ht.name as home_team, at.name as away_team, 
                   g.venue, g.game_time, g.status
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.date >= CURRENT_DATE
            ORDER BY g.date ASC, g.game_time ASC
            LIMIT 20
        """)
        
        games = cur.fetchall()
        
        if not games:
            print("No upcoming games found in the database.")
            return
        
        print("\nUPCOMING GAMES:")
        print(f"{'Date':<12} {'Time':<8} {'Away':<20} {'Home':<20} {'Venue':<30} {'Status':<10}")
        print("-" * 100)
        
        for game in games:
            print(f"{game[1]!s:<12} {game[5]:<8} {game[3]:<20} {game[2]:<20} {game[4]:<30} {game[6]:<10}")
        
    except Exception as e:
        logger.error(f"Error viewing game schedule: {e}")
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

def view_prop_odds(config):
    """View prop betting odds from the database"""
    conn = connect_to_db(config)
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # Get upcoming games
        cur.execute("""
            SELECT g.game_id, g.date, ht.name as home_team, at.name as away_team
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.date >= CURRENT_DATE
            ORDER BY g.date ASC
            LIMIT 10
        """)
        
        games = cur.fetchall()
        
        if not games:
            print("No upcoming games found in the database.")
            return
        
        print("\nSELECT A GAME:")
        for i, game in enumerate(games, 1):
            print(f"{i}. {game[1]} - {game[3]} @ {game[2]}")
        
        game_choice = input("\nEnter game number (or 0 to cancel): ")
        if game_choice == "0":
            return
        
        try:
            game_idx = int(game_choice) - 1
            if game_idx < 0 or game_idx >= len(games):
                print("Invalid game number.")
                return
            
            game_id = games[game_idx][0]
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            return
        
        # Get prop types
        cur.execute("""
            SELECT prop_type_id, name, category
            FROM prop_types
            ORDER BY category, name
        """)
        
        prop_types = cur.fetchall()
        
        if not prop_types:
            print("No prop types found in the database.")
            return
        
        print("\nSELECT A PROP TYPE:")
        for i, prop in enumerate(prop_types, 1):
            print(f"{i}. {prop[1]} ({prop[2]})")
        
        prop_choice = input("\nEnter prop type number (or 0 to cancel): ")
        if prop_choice == "0":
            return
        
        try:
            prop_idx = int(prop_choice) - 1
            if prop_idx < 0 or prop_idx >= len(prop_types):
                print("Invalid prop type number.")
                return
            
            prop_type_id = prop_types[prop_idx][0]
            prop_name = prop_types[prop_idx][1]
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            return
        
        # Get odds for the selected game and prop type
        cur.execute("""
            SELECT p.first_name, p.last_name, s.name as sportsbook, o.line, o.over_odds, o.under_odds
            FROM odds o
            JOIN players p ON o.player_id = p.player_id
            JOIN sportsbooks s ON o.sportsbook_id = s.sportsbook_id
            WHERE o.game_id = %s AND o.prop_type_id = %s
            ORDER BY p.last_name, p.first_name, s.name
        """, (game_id, prop_type_id))
        
        odds = cur.fetchall()
        
        if not odds:
            print(f"No odds found for {prop_name} in this game.")
            return
        
        print(f"\nODDS FOR {prop_name.upper()} - {games[game_idx][3]} @ {games[game_idx][2]} ({games[game_idx][1]})")
        print(f"{'Player':<25} {'Sportsbook':<15} {'Line':<8} {'Over':<8} {'Under':<8}")
        print("-" * 70)
        
        current_player = ""
        for odd in odds:
            player_name = f"{odd[0]} {odd[1]}"
            if player_name != current_player:
                print("-" * 70)
                current_player = player_name
            
            over_odds = f"+{odd[4]}" if odd[4] > 0 else odd[4]
            under_odds = f"+{odd[5]}" if odd[5] > 0 else odd[5]
            
            print(f"{player_name:<25} {odd[2]:<15} {odd[3]:<8.1f} {over_odds:<8} {under_odds:<8}")
        
    except Exception as e:
        logger.error(f"Error viewing prop odds: {e}")
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

def import_data_from_csv(config):
    """Import data from a CSV file"""
    print("\nIMPORT DATA FROM CSV")
    print("This feature is not yet implemented.")
    input("Press Enter to continue...")

def import_example_dataframe(config):
    """Import data from the example dataframe"""
    print("\nIMPORT EXAMPLE DATAFRAME")
    print("Importing data from the Example_Dataframe_May_5 file...")
    
    # Ask about resetting the database
    reset_db = input("Reset database tables before importing? (y/n): ").lower() == 'y'
    
    if reset_db:
        print("Resetting database tables...")
        success = reset_tables(config)
        if not success:
            print("Failed to reset database tables. Aborting import.")
            input("Press Enter to continue...")
            return
    
    # Run the import function
    try:
        success = import_dataframe_to_db(config)
        if success:
            print("\nData imported successfully!")
        else:
            print("\nFailed to import data. Check the logs for details.")
    except Exception as e:
        logger.error(f"Error importing example dataframe: {e}")
        print(f"\nError: {e}")
    
    input("Press Enter to continue...")

def export_predictions_to_csv(config):
    """Export predictions to a CSV file"""
    conn = connect_to_db(config)
    if not conn:
        return
    
    try:
        import pandas as pd
        import os
        from datetime import datetime
        
        cur = conn.cursor()
        
        # Get predictions
        cur.execute("""
            SELECT 
                g.date, 
                ht.name as home_team, 
                at.name as away_team,
                p.first_name || ' ' || p.last_name as player_name,
                pt.name as prop_type,
                pr.predicted_value,
                pr.predicted_probability,
                pr.confidence_level,
                pr.ev_over,
                pr.ev_under
            FROM predictions pr
            JOIN games g ON pr.game_id = g.game_id
            JOIN players p ON pr.player_id = p.player_id
            JOIN prop_types pt ON pr.prop_type_id = pt.prop_type_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.date >= CURRENT_DATE
            ORDER BY g.date, player_name, pt.name
        """)
        
        predictions = cur.fetchall()
        
        if not predictions:
            print("No predictions found in the database.")
            return
        
        # Create a DataFrame
        df = pd.DataFrame(predictions, columns=[
            'Date', 'HomeTeam', 'AwayTeam', 'Player', 'PropType', 
            'PredictedValue', 'Probability', 'Confidence', 'EV_Over', 'EV_Under'
        ])
        
        # Create the data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(data_dir, f"predictions_{timestamp}.csv")
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Predictions exported to {filename}")
        
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

def view_database_status(config):
    """View the status of the database"""
    conn = connect_to_db(config)
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # Get table counts
        tables = [
            'teams', 'players', 'player_batting_stats', 'player_pitching_stats',
            'games', 'prop_types', 'sportsbooks', 'odds', 'predictions'
        ]
        
        print("\nDATABASE STATUS:")
        print("-" * 40)
        
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"{table:<25} {count:>10} rows")
        
        print("-" * 40)
        
        # Get database size
        cur.execute("""
            SELECT pg_size_pretty(pg_database_size(current_database()))
        """)
        db_size = cur.fetchone()[0]
        print(f"Database Size: {db_size}")
        
        # Get last updated timestamps
        print("\nLAST UPDATED:")
        print("-" * 40)
        
        cur.execute("""
            SELECT 'player_batting_stats', MAX(date) FROM player_batting_stats
            UNION ALL
            SELECT 'player_pitching_stats', MAX(date) FROM player_pitching_stats
            UNION ALL
            SELECT 'odds', MAX(timestamp::date) FROM odds
            UNION ALL
            SELECT 'predictions', MAX(timestamp::date) FROM predictions
        """)
        
        timestamps = cur.fetchall()
        for ts in timestamps:
            if ts[1]:
                print(f"{ts[0]:<25} {ts[1]}")
            else:
                print(f"{ts[0]:<25} Never")
        
    except Exception as e:
        logger.error(f"Error viewing database status: {e}")
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

def main():
    """Main function to run the MLB prop betting application"""
    config = load_config()
    
    while True:
        choice = display_menu()
        
        if choice == "0":
            print("Exiting application. Goodbye!")
            break
        
        elif choice == "1":
            # Data Management
            while True:
                data_choice = display_data_menu()
                
                if data_choice == "0":
                    break
                elif data_choice == "1":
                    view_player_stats(config)
                elif data_choice == "2":
                    view_game_schedule(config)
                elif data_choice == "3":
                    view_prop_odds(config)
                elif data_choice == "4":
                    import_data_from_csv(config)
                elif data_choice == "5":
                    import_example_dataframe(config)
                else:
                    print("Invalid choice. Please try again.")
        
        elif choice == "2":
            # Analysis & Predictions
            while True:
                analysis_choice = display_analysis_menu()
                
                if analysis_choice == "0":
                    break
                elif analysis_choice == "1":
                    print("Generating predictions...")
                    generate_predictions(config)
                elif analysis_choice == "2":
                    print("Finding value bets...")
                    find_value_bets(config)
                elif analysis_choice == "3":
                    export_predictions_to_csv(config)
                else:
                    print("Invalid choice. Please try again.")
        
        elif choice == "3":
            # Database Management
            while True:
                db_choice = display_db_menu()
                
                if db_choice == "0":
                    break
                elif db_choice == "1":
                    view_database_status(config)
                elif db_choice == "2":
                    confirm = input("WARNING: This will delete all data in the database. Are you sure? (yes/no): ")
                    if confirm.lower() == "yes":
                        print("Resetting database tables...")
                        reset_tables(config)
                    else:
                        print("Database reset cancelled.")
                else:
                    print("Invalid choice. Please try again.")
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
