#!/usr/bin/env python3
"""
MLB Prop Betting Application
----------------------------
A tool for collecting MLB data, analyzing player statistics, 
and generating prop betting predictions.
"""

import sys
from config import logger, load_config
from db_utils import reset_tables
from team_player_data import fetch_mlb_teams, fetch_players
from stats_games import fetch_player_stats, fetch_game_schedule
from betting import fetch_prop_betting_odds, generate_predictions, find_value_bets

def display_menu():
    """Display the main menu options"""
    print("\nMLB Prop Betting System")
    print("=" * 50)
    print("1. Data Collection")
    print("2. Analysis & Predictions")
    print("3. Run Complete Pipeline")
    print("4. Database Management")
    print("5. Exit")
    print("=" * 50)

def data_collection_menu():
    """Display the data collection sub-menu"""
    print("\nData Collection")
    print("=" * 50)
    print("1. Fetch teams")
    print("2. Fetch players (standard method)")
    print("3. Fetch players (alternative method)")
    print("4. Fetch player statistics")
    print("5. Fetch game schedule")
    print("6. Fetch prop betting odds")
    print("7. Return to main menu")
    print("=" * 50)

def analysis_menu():
    """Display the analysis sub-menu"""
    print("\nAnalysis & Predictions")
    print("=" * 50)
    print("1. Generate predictions")
    print("2. Find value bets")
    print("3. Return to main menu")
    print("=" * 50)

def database_menu():
    """Display the database management sub-menu"""
    print("\nDatabase Management")
    print("=" * 50)
    print("1. Reset database tables (CAUTION!)")
    print("2. Return to main menu")
    print("=" * 50)

def handle_data_collection(config):
    """Handle data collection menu options"""
    data_collection_menu()
    try:
        choice = int(input("\nEnter your choice (1-7): "))
        
        if choice == 1:
            fetch_mlb_teams(config)
        elif choice == 2:
            team_input = input("Enter team ID (leave blank for all teams): ")
            team_id = int(team_input) if team_input.strip() else None
            fetch_players(config, team_id)
        elif choice == 3:
            team_input = input("Enter team ID (leave blank for all teams): ")
            team_id = int(team_input) if team_input.strip() else None
            fetch_players(config, team_id, use_alternative=True)
        elif choice == 4:
            days_back = int(input("Days of stats to fetch (default 30): ") or "30")
            fetch_player_stats(config, days_back)
        elif choice == 5:
            days_forward = int(input("Days forward to fetch (default 7): ") or "7")
            days_back = int(input("Days back to fetch (default 7): ") or "7")
            fetch_game_schedule(config, days_forward, days_back)
        elif choice == 6:
            fetch_prop_betting_odds(config)
        elif choice == 7:
            return
        else:
            print("Invalid choice. Please enter 1-7.")
    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An error occurred. Check the log file for details.")

def handle_analysis(config):
    """Handle analysis menu options"""
    analysis_menu()
    try:
        choice = int(input("\nEnter your choice (1-3): "))
        
        if choice == 1:
            generate_predictions(config)
        elif choice == 2:
            edge = float(input("Minimum edge to look for (default 0.1): ") or "0.1")
            find_value_bets(config, edge)
        elif choice == 3:
            return
        else:
            print("Invalid choice. Please enter 1-3.")
    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An error occurred. Check the log file for details.")

def handle_database(config):
    """Handle database menu options"""
    database_menu()
    try:
        choice = int(input("\nEnter your choice (1-2): "))
        
        if choice == 1:
            confirm = input("This will DELETE all existing data! Type 'YES' to confirm: ")
            if confirm == "YES":
                reset_tables(config)
            else:
                print("Reset cancelled.")
        elif choice == 2:
            return
        else:
            print("Invalid choice. Please enter 1-2.")
    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An error occurred. Check the log file for details.")

def run_complete_pipeline(config):
    """Run the complete data pipeline"""
    print("\nRunning complete pipeline...")
    
    try:
        print("\n1. Fetching teams...")
        fetch_mlb_teams(config)
        
        print("\n2. Fetching players...")
        fetch_players(config)
        
        print("\n3. Fetching player statistics...")
        fetch_player_stats(config)
        
        print("\n4. Fetching game schedule...")
        fetch_game_schedule(config, 7, 1)
        
        print("\n5. Fetching prop betting odds...")
        fetch_prop_betting_odds(config)
        
        print("\n6. Generating predictions...")
        generate_predictions(config)
        
        print("\n7. Finding value bets...")
        find_value_bets(config)
        
        print("\nPipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        print(f"Pipeline failed. Check the log file for details.")

def main():
    """Main function to run the MLB prop betting application"""
    logger.info("MLB Prop Betting Application starting")
    config = load_config()
    
    while True:
        display_menu()
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            
            if choice == 1:
                handle_data_collection(config)
            elif choice == 2:
                handle_analysis(config)
            elif choice == 3:
                run_complete_pipeline(config)
            elif choice == 4:
                handle_database(config)
            elif choice == 5:
                print("Exiting program.")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1-5.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            logger.info("Program terminated by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            print(f"An error occurred. Check the log file for details.")

if __name__ == "__main__":
    main()
