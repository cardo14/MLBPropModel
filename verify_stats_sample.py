#!/usr/bin/env python3
"""
Sample SQL Queries to Verify Database Statistics
-----------------------------------------------
This script contains various SQL queries to verify that all player statistics
are properly stored in the database.
"""

import psycopg2
import configparser
from datetime import datetime

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

def run_verification_queries():
    """Run comprehensive verification queries"""
    config = load_config()
    conn = psycopg2.connect(
        host=config['db_host'],
        database=config['db_name'],
        user=config['db_user'],
        password=config['db_password']
    )
    cur = conn.cursor()
    
    print("=" * 70)
    print("COMPREHENSIVE DATABASE STATISTICS VERIFICATION")
    print("=" * 70)
    
    # Query 1: Overall counts by date range
    print("\n1. OVERALL STATISTICS COUNT BY DATE RANGE:")
    print("-" * 50)
    
    query1 = """
    SELECT 
        'Games' as table_name,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as total_records
    FROM games
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    UNION ALL
    
    SELECT 
        'Batting Stats' as table_name,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as total_records
    FROM player_batting_stats
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    UNION ALL
    
    SELECT 
        'Pitching Stats' as table_name,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as total_records
    FROM player_pitching_stats
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    ORDER BY table_name;
    """
    
    cur.execute(query1)
    results = cur.fetchall()
    
    print(f"{'Table':<15} {'Min Date':<12} {'Max Date':<12} {'Records':<10}")
    print("-" * 50)
    for table, min_date, max_date, count in results:
        print(f"{table:<15} {str(min_date):<12} {str(max_date):<12} {count:<10}")
    
    # Query 2: Daily statistics breakdown
    print("\n2. DAILY STATISTICS BREAKDOWN (Sample - Last 10 Days with Data):")
    print("-" * 65)
    
    query2 = """
    SELECT 
        g.date,
        COUNT(g.game_id) as games,
        COALESCE(b.batting_count, 0) as batting_stats,
        COALESCE(p.pitching_count, 0) as pitching_stats,
        CASE 
            WHEN COALESCE(b.batting_count, 0) > 0 AND COALESCE(p.pitching_count, 0) > 0 
            THEN 'Complete' 
            ELSE 'Missing' 
        END as status
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
    WHERE g.date >= '2025-04-27' AND g.date <= '2025-05-28'
    GROUP BY g.date, b.batting_count, p.pitching_count
    ORDER BY g.date DESC
    LIMIT 10;
    """
    
    cur.execute(query2)
    results = cur.fetchall()
    
    print(f"{'Date':<12} {'Games':<6} {'Batting':<8} {'Pitching':<9} {'Status':<10}")
    print("-" * 65)
    for date, games, batting, pitching, status in results:
        print(f"{str(date):<12} {games:<6} {batting:<8} {pitching:<9} {status:<10}")
    
    # Query 3: Sample player statistics for a specific date
    print("\n3. SAMPLE PLAYER BATTING STATS (May 28, 2025 - Top 10 by At-Bats):")
    print("-" * 70)
    
    query3 = """
    SELECT 
        p.first_name || ' ' || p.last_name as player_name,
        t.name as team,
        bs.at_bats,
        bs.hits,
        bs.runs,
        bs.home_runs,
        bs.rbi,
        ROUND(bs.batting_average::numeric, 3) as avg
    FROM player_batting_stats bs
    JOIN players p ON bs.player_id = p.player_id
    JOIN teams t ON p.team_id = t.team_id
    WHERE bs.date = '2025-05-28'
    ORDER BY bs.at_bats DESC, bs.hits DESC
    LIMIT 10;
    """
    
    cur.execute(query3)
    results = cur.fetchall()
    
    print(f"{'Player':<20} {'Team':<15} {'AB':<3} {'H':<3} {'R':<3} {'HR':<3} {'RBI':<4} {'AVG':<6}")
    print("-" * 70)
    for player, team, ab, h, r, hr, rbi, avg in results:
        print(f"{player:<20} {team:<15} {ab:<3} {h:<3} {r:<3} {hr:<3} {rbi:<4} {avg:<6}")
    
    # Query 4: Sample pitching statistics for a specific date
    print("\n4. SAMPLE PLAYER PITCHING STATS (May 28, 2025 - Top 10 by Innings):")
    print("-" * 75)
    
    query4 = """
    SELECT 
        p.first_name || ' ' || p.last_name as player_name,
        t.name as team,
        ps.innings_pitched,
        ps.hits_allowed,
        ps.runs_allowed,
        ps.earned_runs,
        ps.strikeouts,
        ROUND(ps.era::numeric, 2) as era
    FROM player_pitching_stats ps
    JOIN players p ON ps.player_id = p.player_id
    JOIN teams t ON p.team_id = t.team_id
    WHERE ps.date = '2025-05-28'
    ORDER BY ps.innings_pitched DESC, ps.strikeouts DESC
    LIMIT 10;
    """
    
    cur.execute(query4)
    results = cur.fetchall()
    
    print(f"{'Player':<20} {'Team':<15} {'IP':<5} {'H':<3} {'R':<3} {'ER':<3} {'K':<3} {'ERA':<6}")
    print("-" * 75)
    for player, team, ip, h, r, er, k, era in results:
        print(f"{player:<20} {team:<15} {ip:<5} {h:<3} {r:<3} {er:<3} {k:<3} {era:<6}")
    
    # Query 5: Verify data completeness for specific date ranges
    print("\n5. DATA COMPLETENESS CHECK BY WEEK:")
    print("-" * 50)
    
    query5 = """
    SELECT 
        DATE_TRUNC('week', g.date)::date as week_start,
        COUNT(DISTINCT g.date) as game_days,
        COUNT(g.game_id) as total_games,
        COUNT(DISTINCT bs.date) as days_with_batting,
        COUNT(DISTINCT ps.date) as days_with_pitching,
        SUM(CASE WHEN bs.date IS NOT NULL THEN 1 ELSE 0 END) as batting_records,
        SUM(CASE WHEN ps.date IS NOT NULL THEN 1 ELSE 0 END) as pitching_records
    FROM games g
    LEFT JOIN player_batting_stats bs ON g.date = bs.date
    LEFT JOIN player_pitching_stats ps ON g.date = ps.date
    WHERE g.date >= '2025-04-27' AND g.date <= '2025-05-28'
    GROUP BY DATE_TRUNC('week', g.date)
    ORDER BY week_start;
    """
    
    cur.execute(query5)
    results = cur.fetchall()
    
    print(f"{'Week Start':<12} {'Days':<5} {'Games':<6} {'Bat Days':<9} {'Pitch Days':<10} {'Batting':<8} {'Pitching':<9}")
    print("-" * 70)
    for week, days, games, bat_days, pitch_days, batting, pitching in results:
        print(f"{str(week):<12} {days:<5} {games:<6} {bat_days:<9} {pitch_days:<10} {batting:<8} {pitching:<9}")
    
    # Query 6: Check for any remaining gaps
    print("\n6. FINAL GAP CHECK - Dates with Games but Missing Stats:")
    print("-" * 55)
    
    query6 = """
    SELECT 
        g.date,
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
    WHERE g.date >= '2025-04-27' AND g.date <= '2025-05-28'
    AND (COALESCE(b.batting_count, 0) = 0 OR COALESCE(p.pitching_count, 0) = 0)
    GROUP BY g.date, b.batting_count, p.pitching_count
    ORDER BY g.date;
    """
    
    cur.execute(query6)
    results = cur.fetchall()
    
    if results:
        print("FOUND GAPS:")
        print(f"{'Date':<12} {'Games':<6} {'Batting':<8} {'Pitching':<8}")
        print("-" * 35)
        for date, games, batting, pitching in results:
            print(f"{str(date):<12} {games:<6} {batting:<8} {pitching:<8}")
    else:
        print("✅ NO GAPS FOUND! All dates have complete statistics.")
    
    # Query 7: Summary statistics
    print("\n7. SUMMARY STATISTICS:")
    print("-" * 25)
    
    query7 = """
    SELECT 
        'Total Games (Apr 27 - May 28)' as metric,
        COUNT(*)::text as value
    FROM games 
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    UNION ALL
    
    SELECT 
        'Total Batting Records' as metric,
        COUNT(*)::text as value
    FROM player_batting_stats 
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    UNION ALL
    
    SELECT 
        'Total Pitching Records' as metric,
        COUNT(*)::text as value
    FROM player_pitching_stats 
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    UNION ALL
    
    SELECT 
        'Unique Players with Batting Stats' as metric,
        COUNT(DISTINCT player_id)::text as value
    FROM player_batting_stats 
    WHERE date >= '2025-04-27' AND date <= '2025-05-28'
    
    UNION ALL
    
    SELECT 
        'Unique Players with Pitching Stats' as metric,
        COUNT(DISTINCT player_id)::text as value
    FROM player_pitching_stats 
    WHERE date >= '2025-04-27' AND date <= '2025-05-28';
    """
    
    cur.execute(query7)
    results = cur.fetchall()
    
    for metric, value in results:
        print(f"{metric:<35}: {value}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE!")
    print("=" * 70)
    
    conn.close()

if __name__ == "__main__":
    run_verification_queries() 