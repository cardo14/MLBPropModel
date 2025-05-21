import requests
import time
import psycopg2
import random
from datetime import datetime, timedelta
from config import logger
from db_utils import connect_to_db


def fetch_player_stats(config, days_back=30, batch_size=20, delay_between_batches=3):
    """Fetch recent player statistics with batching for better reliability"""
    logger.info(
        f"Fetching player statistics for the last {days_back} days in batches of {batch_size}...")

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
        batches = [all_players[i:i + batch_size]
                   for i in range(0, total_players, batch_size)]

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Processing batch {batch_num+1}/{len(batches)} ({len(batch)} players)")
            batch_stats_added = 0

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
                        logger.warning(
                            f"API returned status {response.status_code} for player {player_name}")
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

                            batch_stats_added += 1

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

                            batch_stats_added += 1

                except Exception as e:
                    logger.error(
                        f"Error processing stats for player {player_name}: {e}")
                    continue

                # Brief delay between player requests to avoid rate limiting
                time.sleep(0.2)

            # Commit after each batch
            conn.commit()
            stats_added += batch_stats_added
            logger.info(
                f"Completed batch {batch_num+1}/{len(batches)}, added {batch_stats_added} stats in this batch")

            # Sleep between batches to avoid overwhelming the API
            if batch_num < len(batches) - 1:
                logger.info(
                    f"Sleeping for {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)

        logger.info(
            f"Player stats imported! Added/updated {stats_added} stat records.")
        return stats_added

    except Exception as e:
        logger.error(f"Error in fetch_player_stats: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def fetch_game_schedule(config, days_forward=7, days_back=7):
    """Fetch MLB game schedule and results"""
    logger.info(
        f"Fetching game schedule ({days_back} days back, {days_forward} days forward)...")

    start_date = (datetime.now() - timedelta(days=days_back)
                  ).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_forward)
                ).strftime("%Y-%m-%d")

    url = f"{config['api_base_url']}/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",
        "hydrate": "team,venue"
    }

    conn = connect_to_db(config)
    if not conn:
        return 0

    cur = conn.cursor()
    games_added = 0

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "dates" not in data:
            logger.error("No game data returned from API")
            return 0

        # Process each date in the schedule
        for date_data in data["dates"]:
            game_date = date_data["date"]

            if "games" not in date_data:
                continue

            # Process each game on this date
            for game in date_data["games"]:
                game_id = game.get("gamePk")
                if not game_id:
                    continue

                away_team_id = game.get("teams", {}).get(
                    "away", {}).get("team", {}).get("id")
                home_team_id = game.get("teams", {}).get(
                    "home", {}).get("team", {}).get("id")
                venue = game.get("venue", {}).get("name", "")
                game_time = game.get("gameDate", "").split(
                    "T")[1][:5] if "T" in game.get("gameDate", "") else ""
                status = game.get("status", {}).get("detailedState", "")

                # Determine if the venue is a dome
                venue_is_dome = False
                if "roof" in venue.lower() or "dome" in venue.lower():
                    venue_is_dome = True

                # Insert game data
                cur.execute("""
                    INSERT INTO games
                    (game_id, date, home_team_id, away_team_id, venue, venue_is_dome, game_time, status)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id) DO UPDATE
                    SET date = EXCLUDED.date,
                        home_team_id = EXCLUDED.home_team_id,
                        away_team_id = EXCLUDED.away_team_id,
                        venue = EXCLUDED.venue,
                        venue_is_dome = EXCLUDED.venue_is_dome,
                        game_time = EXCLUDED.game_time,
                        status = EXCLUDED.status
                """, (game_id, game_date, home_team_id, away_team_id, venue, venue_is_dome, game_time, status))

                games_added += 1

        conn.commit()
        logger.info(
            f"Game schedule imported! Added/updated {games_added} games.")
        return games_added

    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching game schedule: {e}")
    except Exception as e:
        logger.error(f"Error in fetch_game_schedule: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def get_player_id_by_name(config, first_name, last_name):
    """Return player_id from database given player's first and last name."""
    conn = connect_to_db(config)
    if not conn:
        return None

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT player_id FROM players 
            WHERE LOWER(first_name) = LOWER(%s) AND LOWER(last_name) = LOWER(%s)
        """, (first_name, last_name))

        result = cur.fetchone()
        return result[0] if result else None

    except Exception as e:
        logger.error(
            f"Error fetching player_id for {first_name} {last_name}: {e}")
        return None

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
