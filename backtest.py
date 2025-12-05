import pickle
import numpy as np
import pandas as pd
import sqlite3
from xgboost import XGBClassifier


def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


def american_to_decimal(ml):
    return 1 + (ml / 100 if ml > 0 else 100 / abs(ml))


def implied_prob(ml):
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def simulate_bets(df, model):
    bank = 1000
    history = []

    for _, row in df.iterrows():
        X = row[["HOME_PTS_ROLL","AWAY_PTS_ROLL"]].values.reshape(1, -1)
        p = model.predict_proba(X)[0][1]

        ml_home = row["HOME_LINE"]
        d = american_to_decimal(ml_home)

        ev = p * (d - 1) - (1 - p)

        if ev > 0:
            stake = bank * 0.02
            if row["HOME_WIN"] == 1:
                bank += stake * (d - 1)
            else:
                bank -= stake

        history.append(bank)

    return pd.Series(history)


if __name__ == "__main__":
    conn = sqlite3.connect("nba.db")
    df = pd.read_sql("SELECT * FROM games", conn)
    conn.close()

    df["HOME_LINE"] = -110
    df["AWAY_LINE"] = -110

    model = load_model()
    res = simulate_bets(df, model)

    print("\nFinal bankroll:", res.iloc[-1])
    print("ROI:", (res.iloc[-1] - 1000) / 1000)
