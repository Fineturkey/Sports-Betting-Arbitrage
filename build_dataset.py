import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.library.parameters import SeasonAll
import sqlite3
import os

# ========================
# CONFIG
# ========================

START_SEASON = 2003   # 2003–04 season
END_SEASON = 2024     # up to 2023–24
DB_FILE = "nba.db"

# ========================
# Helpers
# ========================

def season_str(year):
    """Return season format like '2003-04'"""
    return f"{year}-{str(year+1)[-2:]}"

def save_to_db(df, table):
    conn = sqlite3.connect(DB_FILE)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

# ========================
# MAIN
# ========================

all_games = []

print(f"Downloading games from {START_SEASON} to {END_SEASON}...")
for year in tqdm(range(START_SEASON, END_SEASON + 1)):
    season = season_str(year)

    # Fetch all games for that season
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = finder.get_data_frames()[0]

    # Keep only regular season + playoffs
    df = df[df["SEASON_ID"].astype(str).str.contains(str(year))]

    all_games.append(df)

games = pd.concat(all_games, ignore_index=True)
games = games.drop_duplicates(subset=["GAME_ID"])

print(f"Total games downloaded: {len(games)}")

# ===================================
# BOX SCORES (team statistics)
# ===================================

box_rows = []
print("Downloading box scores (will take time)...")

for gid in tqdm(games["GAME_ID"].unique()):
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
        df = box.get_data_frames()[0]

        # Team total rows only
        df = df[df["TEAM_ID"].notna()]
        df["GAME_ID"] = gid
        box_rows.append(df)
        time.sleep(0.6)  # throttle to avoid rate limit
    except:
        continue

box = pd.concat(box_rows, ignore_index=True)

# ===================================
# Merge game info + statistics
# ===================================

print("Merging datasets...")

merged = games.merge(
    box,
    on="GAME_ID",
    suffixes=("", "_BOX")
)

# Keep only relevant columns
keep_cols = [
    "GAME_ID", "GAME_DATE", "SEASON_ID",
    "TEAM_ID", "TEAM_ABBREVIATION",
    "WL", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
]

merged = merged[keep_cols]

# Format dates
merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])

# Save to CSV
merged.to_csv("games.csv", index=False)
print("Saved games.csv!")

# Save to SQLite
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

print("Saving to SQLite database...")
conn = sqlite3.connect(DB_FILE)
merged.to_sql("games", conn, index=False)
conn.close()

print("ALL DONE: games.csv + nba.db built successfully!")
