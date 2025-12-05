import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle


def load_data():
    conn = sqlite3.connect("nba.db")
    df = pd.read_sql("SELECT * FROM games", conn)
    conn.close()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def build_team_features(df):
    # sort by team + date so rolling stats make sense
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])

    rolling_features = [
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FG_PCT", "FG3_PCT", "FT_PCT"
    ]

    # 10-game rolling averages per team
    for f in rolling_features:
        df[f"{f}_ROLL"] = (
            df.groupby("TEAM_ID")[f]
              .rolling(10)
              .mean()
              .reset_index(level=0, drop=True)
        )

    # target: did this team win?
    df["WIN"] = (df["WL"] == "W").astype(int)

    # drop early games that don't have a 10-game history
    df = df.dropna(subset=[f"{f}_ROLL" for f in rolling_features])

    return df


def make_matchups(df):
    """
    Build one row per GAME_ID with home + away rolling stats.
    Robust to weird games: if we can't find exactly one home and one away,
    we skip that game.
    """
    games = []

    # group by game id
    for gid, g in df.groupby("GAME_ID"):
        # safety: MATCHUP can be NaN
        matchups = g["MATCHUP"].fillna("")

        # NBA convention: home has "vs", away has "@"
        home_mask = matchups.str.contains("vs", case=False, regex=False)
        away_mask = matchups.str.contains("@", case=False, regex=False)

        g_home = g[home_mask]
        g_away = g[away_mask]

        # If we don't have exactly 1 home and 1 away, skip the game
        if len(g_home) != 1 or len(g_away) != 1:
            # You can uncomment this to see skipped games:
            # print(f"[WARN] Skipping GAME_ID {gid}: home={len(g_home)}, away={len(g_away)}")
            continue

        home = g_home.iloc[0]
        away = g_away.iloc[0]

        row = {
            "GAME_ID": gid,
            "HOME_TEAM": home["TEAM_ID"],
            "AWAY_TEAM": away["TEAM_ID"],
            "HOME_WIN": home["WIN"],
        }

        # which rolling features to include
        feats = [
            "PTS_ROLL", "REB_ROLL", "AST_ROLL", "STL_ROLL", "BLK_ROLL",
            "TOV_ROLL", "FG_PCT_ROLL", "FG3_PCT_ROLL", "FT_PCT_ROLL"
        ]

        for f in feats:
            row[f"HOME_{f}"] = home[f]
            row[f"AWAY_{f}"] = away[f]

        games.append(row)

    matchups = pd.DataFrame(games)
    print(f"Built {len(matchups)} matchups from {df['GAME_ID'].nunique()} unique games.")
    return matchups


def train_model():
    df = load_data()
    print(f"Loaded {len(df)} team-game rows from DB")

    df = build_team_features(df)
    print(f"After rolling features & dropna: {len(df)} rows")

    matchups = make_matchups(df)
