#!/usr/bin/env python3
"""
Underdog Scraper for MLB Prop Betting Lines
Scrapes prop betting lines for walks and strikeouts from Underdog Fantasy
"""

import requests
import json
import pandas as pd
import time
from datetime import datetime, date
import os
import sys

class UnderdogScraper:
    def __init__(self):
        self.base_url = "https://api.underdognetwork.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_mlb_games(self, date_str=None):
        """Get MLB games for a specific date"""
        if date_str is None:
            date_str = date.today().strftime('%Y-%m-%d')
        
        try:
            # This is a placeholder URL - actual API endpoints would need to be determined
            url = f"{self.base_url}/v1/games?sport=MLB&date={date_str}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching games: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error in get_mlb_games: {e}")
            return None
    
    def get_player_props(self, game_id):
        """Get player props for a specific game"""
        try:
            # This is a placeholder URL - actual API endpoints would need to be determined
            url = f"{self.base_url}/v1/games/{game_id}/props"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching props: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error in get_player_props: {e}")
            return None
    
    def extract_walks_strikeouts_props(self, props_data):
        """Extract walks and strikeouts props from the props data"""
        walks_props = []
        strikeouts_props = []
        
        if not props_data:
            return walks_props, strikeouts_props
        
        # This would need to be adapted based on actual API response structure
        for prop in props_data.get('props', []):
            if 'walk' in prop.get('name', '').lower():
                walks_props.append({
                    'player_id': prop.get('player_id'),
                    'player_name': prop.get('player_name'),
                    'line': prop.get('line'),
                    'over_odds': prop.get('over_odds'),
                    'under_odds': prop.get('under_odds'),
                    'stat_type': 'walks'
                })
            elif 'strikeout' in prop.get('name', '').lower():
                strikeouts_props.append({
                    'player_id': prop.get('player_id'),
                    'player_name': prop.get('player_name'),
                    'line': prop.get('line'),
                    'over_odds': prop.get('over_odds'),
                    'under_odds': prop.get('under_odds'),
                    'stat_type': 'strikeouts'
                })
        
        return walks_props, strikeouts_props
    
    def scrape_daily_props(self, date_str=None):
        """Scrape all MLB props for a given date"""
        print(f"Scraping props for date: {date_str or 'today'}")
        
        # Get games for the date
        games = self.get_mlb_games(date_str)
        if not games:
            print("No games found")
            return pd.DataFrame()
        
        all_props = []
        
        # For each game, get the props
        for game in games.get('games', []):
            game_id = game.get('id')
            if game_id:
                print(f"Fetching props for game {game_id}")
                props = self.get_player_props(game_id)
                
                if props:
                    walks_props, strikeouts_props = self.extract_walks_strikeouts_props(props)
                    all_props.extend(walks_props + strikeouts_props)
                
                # Rate limiting
                time.sleep(1)
        
        # Convert to DataFrame
        if all_props:
            df = pd.DataFrame(all_props)
            df['scraped_at'] = datetime.now()
            df['game_date'] = date_str or date.today().strftime('%Y-%m-%d')
            return df
        else:
            return pd.DataFrame()
    
    def save_props_to_csv(self, df, filename=None):
        """Save props data to CSV file"""
        if filename is None:
            date_str = date.today().strftime('%Y%m%d')
            filename = f"underdog_props_{date_str}.csv"
        
        filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"Props saved to {filepath}")

def main():
    """Main function to run the scraper"""
    scraper = UnderdogScraper()
    
    # Scrape today's props
    props_df = scraper.scrape_daily_props()
    
    if not props_df.empty:
        print(f"Scraped {len(props_df)} props")
        scraper.save_props_to_csv(props_df)
    else:
        print("No props found")

if __name__ == "__main__":
    main() 