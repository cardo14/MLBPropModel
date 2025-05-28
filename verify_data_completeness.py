#!/usr/bin/env python3
"""
Data Completeness Verification Script
------------------------------------
This script verifies that all games have corresponding player statistics
and identifies any remaining gaps in the database.
"""

import psycopg2
import configparser
from datetime import datetime, timedelta

def load_config():
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return {
        'db_host': config.get('Database', 'host', fallback='localhost'),
        'db_name': config.get('Database', 'database', fallback='mlb_prop_betting'),
        'db_user': config.get('Database', 'user', fallback='postgres'),
        'db_password': config.get('Database', 'password', fallback='ophadke'),
    }

def verify_data_completeness():
    """Comprehensive verification of data completeness"""
    config = load_config()
    conn = psycopg2.connect(
        host=config['db_host'],
        database=config['db_name'],
        user=config['db_user'],
        password=config['db_password']
    )
    cur = conn.cursor()
    
    print("=" * 60)
    print("MLB DATABASE COMPLETENESS VERIFICATION")
    print("=" * 60)
    
    # 1. Overall data summary
    print("\n1. OVERALL DATA SUMMARY:")
    print("-" * 30)
    
    cur.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM games')
    games_info = cur.fetchone()
    print(f"Games: {games_info[0]} to {games_info[1]} ({games_info[2]} total)")
    
    cur.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM player_batting_stats WHERE date IS NOT NULL')
    batting_info = cur.fetchone()
    print(f"Batting Stats: {batting_info[0]} to {batting_info[1]} ({batting_info[2]} total)")
    
    cur.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM player_pitching_stats WHERE date IS NOT NULL')
    pitching_info = cur.fetchone()
    print(f"Pitching Stats: {pitching_info[0]} to {pitching_info[1]} ({pitching_info[2]} total)")
    
    # 2. Check for dates with games but no stats
    print("\n2. DATES WITH GAMES BUT MISSING STATISTICS:")
    print("-" * 45)
    
    cur.execute("""
        SELECT g.date, 
               COUNT(g.game_id) as games,
               COALESCE(b.batting_count, 0) as batting_stats,
               COALESCE(p.pitching_count, 0) as pitching_stats
        FROM games g
        LEFT JOIN (
            SELECT date, COUNT(*) as batting_count 
            FROM player_batting_stats 
            GROUP BY date
        ) b ON g.date = b.date
        LEFT JOIN (
            SELECT date, COUNT(*) as pitching_count 
            FROM player_pitching_stats 
            GROUP BY date
        ) p ON g.date = p.date
        WHERE g.date >= '2025-04-01'  -- Start of season
        GROUP BY g.date, b.batting_count, p.pitching_count
        HAVING COALESCE(b.batting_count, 0) = 0 OR COALESCE(p.pitching_count, 0) = 0
        ORDER BY g.date
    """)
    
    missing_stats = cur.fetchall()
    if missing_stats:
        print("FOUND MISSING STATISTICS:")
        print(f"{'Date':<12} {'Games':<6} {'Batting':<8} {'Pitching':<8}")
        print("-" * 35)
        for date, games, batting, pitching in missing_stats:
            print(f"{date!s:<12} {games:<6} {batting:<8} {pitching:<8}")
    else:
        print("✅ NO MISSING STATISTICS FOUND - All game dates have player stats!")
    
    # 3. Recent games verification (last 10 days)
    print("\n3. RECENT GAMES VERIFICATION (Last 10 Days):")
    print("-" * 45)
    
    cur.execute("""
        SELECT g.date, 
               COUNT(g.game_id) as games,
               COALESCE(b.batting_count, 0) as batting_stats,
               COALESCE(p.pitching_count, 0) as pitching_stats
        FROM games g
        LEFT JOIN (
            SELECT date, COUNT(*) as batting_count 
            FROM player_batting_stats 
            GROUP BY date
        ) b ON g.date = b.date
        LEFT JOIN (
            SELECT date, COUNT(*) as pitching_count 
            FROM player_pitching_stats 
            GROUP BY date
        ) p ON g.date = p.date
        WHERE g.date >= (SELECT MAX(date) - INTERVAL '10 days' FROM games)
        GROUP BY g.date, b.batting_count, p.pitching_count
        ORDER BY g.date DESC
    """)
    
    recent_games = cur.fetchall()
    print(f"{'Date':<12} {'Games':<6} {'Batting':<8} {'Pitching':<8} {'Status':<10}")
    print("-" * 50)
    
    for date, games, batting, pitching in recent_games:
        status = "✅ Complete" if batting > 0 and pitching > 0 else "❌ Missing"
        print(f"{date!s:<12} {games:<6} {batting:<8} {pitching:<8} {status:<10}")
    
    # 4. Monthly summary
    print("\n4. MONTHLY DATA SUMMARY:")
    print("-" * 25)
    
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM g.date) as year,
            EXTRACT(MONTH FROM g.date) as month,
            COUNT(g.game_id) as games,
            COALESCE(SUM(b.batting_count), 0) as total_batting_stats,
            COALESCE(SUM(p.pitching_count), 0) as total_pitching_stats,
            COUNT(DISTINCT g.date) as game_days
        FROM games g
        LEFT JOIN (
            SELECT date, COUNT(*) as batting_count 
            FROM player_batting_stats 
            GROUP BY date
        ) b ON g.date = b.date
        LEFT JOIN (
            SELECT date, COUNT(*) as pitching_count 
            FROM player_pitching_stats 
            GROUP BY date
        ) p ON g.date = p.date
        GROUP BY EXTRACT(YEAR FROM g.date), EXTRACT(MONTH FROM g.date)
        ORDER BY year, month
    """)
    
    monthly_data = cur.fetchall()
    print(f"{'Year-Month':<10} {'Games':<6} {'Days':<5} {'Batting':<8} {'Pitching':<8}")
    print("-" * 45)
    
    for year, month, games, batting, pitching, days in monthly_data:
        month_name = datetime(int(year), int(month), 1).strftime('%Y-%m')
        print(f"{month_name:<10} {games:<6} {days:<5} {batting:<8} {pitching:<8}")
    
    # 5. Data quality checks
    print("\n5. DATA QUALITY CHECKS:")
    print("-" * 25)
    
    # Check for duplicate stats
    cur.execute("""
        SELECT COUNT(*) as duplicates
        FROM (
            SELECT player_id, date, stat_period, COUNT(*)
            FROM player_batting_stats
            GROUP BY player_id, date, stat_period
            HAVING COUNT(*) > 1
        ) dups
    """)
    batting_dups = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(*) as duplicates
        FROM (
            SELECT player_id, date, stat_period, COUNT(*)
            FROM player_pitching_stats
            GROUP BY player_id, date, stat_period
            HAVING COUNT(*) > 1
        ) dups
    """)
    pitching_dups = cur.fetchone()[0]
    
    print(f"Duplicate batting stats: {batting_dups}")
    print(f"Duplicate pitching stats: {pitching_dups}")
    
    # Check for stats without corresponding games
    cur.execute("""
        SELECT COUNT(DISTINCT bs.date) as orphaned_dates
        FROM player_batting_stats bs
        LEFT JOIN games g ON bs.date = g.date
        WHERE g.date IS NULL
    """)
    orphaned_batting = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(DISTINCT ps.date) as orphaned_dates
        FROM player_pitching_stats ps
        LEFT JOIN games g ON ps.date = g.date
        WHERE g.date IS NULL
    """)
    orphaned_pitching = cur.fetchone()[0]
    
    print(f"Batting stats without games: {orphaned_batting} dates")
    print(f"Pitching stats without games: {orphaned_pitching} dates")
    
    # 6. Final summary
    print("\n6. FINAL VERIFICATION SUMMARY:")
    print("-" * 30)
    
    total_game_dates = cur.execute("SELECT COUNT(DISTINCT date) FROM games")
    total_game_dates = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT date) FROM player_batting_stats")
    batting_dates = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT date) FROM player_pitching_stats")
    pitching_dates = cur.fetchone()[0]
    
    print(f"Total game dates: {total_game_dates}")
    print(f"Dates with batting stats: {batting_dates}")
    print(f"Dates with pitching stats: {pitching_dates}")
    
    if batting_dates == total_game_dates and pitching_dates == total_game_dates:
        print("\n🎉 DATABASE IS COMPLETE! All game dates have corresponding player statistics.")
    else:
        print(f"\n⚠️  DATABASE INCOMPLETE:")
        print(f"   Missing batting stats for {total_game_dates - batting_dates} dates")
        print(f"   Missing pitching stats for {total_game_dates - pitching_dates} dates")
    
    conn.close()

if __name__ == "__main__":
    verify_data_completeness() 