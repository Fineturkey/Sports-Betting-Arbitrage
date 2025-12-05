import time
import os
import sqlite3

import pandas as pd
import requests
from tqdm import tqdm
from nba_api.stats.endpoints import leaguegamefinder

START_SEASON = 2003
END_SEASON   = 2024
DB_FILE      = "nba.db"

TIMEOUT      = 30
RETRIES      = 5
SLEEP_BETWEEN = 0.3


def season_str(y): return f"{y}-{str(y+1)[-2:]}"


print(f"Downloading NBA game-level data {START_SEASON}–{END_SEASON} (FAST)…")

all_games = []

for year in tqdm(range(START_SEASON, END_SEASON + 1)):
    season = season_str(year)
    df_season = None

    for attempt in range(RETRIES):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                timeout=TIMEOUT,
                league_id_nullable="00"
            )
            df = finder.get_data_frames()[0]
            df = df[df["SEASON_ID"].astype(str).str.contains(str(year))]
            df_season = df
            break
        except Exception as e:
            print(f"[{season}] retry {attempt+1}/{RETRIES}: {e}")
            time.sleep(1)

    if df_season is None:
        print(f"[{season}] FAILED, skipping.")
        continue

    all_games.append(df_season)
    time.sleep(SLEEP_BETWEEN)


games = pd.concat(all_games, ignore_index=True)
games = games.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])

keep_cols = [
    "GAME_ID", "GAME_DATE", "SEASON_ID",
    "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP",
    "WL", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT"
]

games = games[keep_cols]
games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

games.to_csv("games.csv", index=False)
print("Saved games.csv")

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
games.to_sql("games", conn, index=False)
conn.close()

print("ALL DONE — Database ready!")
