import pandas as pd
import requests
import json
import os
from datetime import datetime


def get_pp_projections():
    # Define the URL and headers as in the curl request
    url = 'https://api.prizepicks.com/projections?league_id=2&per_page=250&single_stat=true&in_game=true&state_code=CA&game_mode=pickem'

    headers = {
        'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://app.prizepicks.com/',
        'X-Device-ID': '1a9d6304-65f3-4304-8523-ccf458d3c0c4',
        'sec-ch-ua-platform': '"macOS"'
    }

    # Send the GET request
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

    opponents = []
    player_ids = []
    stat_type = []
    line = []
    for item in data['data']:
        player_info = item['relationships']
        player_id = player_info['new_player']['data']['id']
        player_ids.append(player_id)
        item = item['attributes']
        opponents.append(item['description'])
        stat_type.append(item['stat_type'])
        line.append(item['line_score'])

    players = []
    for id in player_ids:
        for item in data['included']:
            if item['type'] == 'new_player':
                if item['id'] == id:
                    players.append(item['attributes']['name'])
    return pd.DataFrame({
        'players': players,
        'opponents': opponents,
        'stat_type': stat_type,
        'line': line
    })


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"/Users/leonardocardozo/Python/Sports_Betting_Project/prizepicks_scraper.py_{timestamp}.csv"
    pp_projections = get_pp_projections()
    pp_projections.to_csv(filename, index=False)
    print("Data saved to prizepicks_props.csv")


if __name__ == "__main__":
    main()
