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
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    rolling_features = ["PTS", "REB", "AST", "STL", "BLK", "TOV",
                        "FG_PCT", "FG3_PCT", "FT_PCT"]

    for f in rolling_features:
        df[f"{f}_ROLL"] = df.groupby("TEAM_ID")[f].rolling(10).mean().reset_index(level=0, drop=True)

    df["WIN"] = (df["WL"] == "W").astype(int)
    return df.dropna()


def make_matchups(df):
    games = []
    for gid, g in df.groupby("GAME_ID"):
        if len(g) != 2:
            continue  # safety check

        home = g[g["MATCHUP"].str.contains("vs")].iloc[0]
        away = g[g["MATCHUP"].str.contains("@")].iloc[0]

        row = {
            "GAME_ID": gid,
            "HOME_TEAM": home["TEAM_ID"],
            "AWAY_TEAM": away["TEAM_ID"],
            "HOME_WIN": home["WIN"]
        }

        feats = ["PTS_ROLL","REB_ROLL","AST_ROLL","STL_ROLL","BLK_ROLL",
                 "TOV_ROLL","FG_PCT_ROLL","FG3_PCT_ROLL","FT_PCT_ROLL"]

        for f in feats:
            row[f"HOME_{f}"] = home[f]
            row[f"AWAY_{f}"] = away[f]

        games.append(row)

    return pd.DataFrame(games)


def train_model():
    df = load_data()
    df = build_team_features(df)
    matchups = make_matchups(df)

    X = matchups.drop(["GAME_ID", "HOME_WIN"], axis=1)
    y = matchups["HOME_WIN"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    print("Train Acc:", model.score(X_train, y_train))
    print("Test Acc:", model.score(X_test, y_test))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved model.pkl")


if __name__ == "__main__":
    train_model()
