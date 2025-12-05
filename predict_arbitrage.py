import pickle
import numpy as np


def american_to_decimal(ml):
    ml = int(ml)
    if ml > 0:
        return 1 + ml / 100
    else:
        return 1 + 100 / abs(ml)


def implied_prob(ml):
    ml = int(ml)
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def fair_american(prob):
    dec = 1 / prob
    if dec >= 2:
        return int((dec - 1) * 100)    # dog
    else:
        return int(-100 / (dec - 1))   # favorite


def kelly(prob, ml):
    d = american_to_decimal(ml)
    q = 1 - prob
    b = d - 1
    return max(0, prob - q / b)


def evaluate_bet(prob, ml):
    ip = implied_prob(ml)
    edge = prob - ip
    fair = fair_american(prob)
    dec = american_to_decimal(ml)
    ev = prob * (dec - 1) - (1 - prob)
    k = kelly(prob, ml)

    return {
        "prob": prob,
        "fair_ml": fair,
        "implied_prob": ip,
        "edge": edge,
        "ev": ev,
        "kelly_fraction": k
    }


def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


def predict(home_feats, away_feats):
    model = load_model()

    # Combine into one feature row
    row = {}
    for k, v in home_feats.items():
        row[f"HOME_{k}"] = v
    for k, v in away_feats.items():
        row[f"AWAY_{k}"] = v

    X = np.array([list(row.values())], dtype=float)
    p = model.predict_proba(X)[0][1]  # probability home wins
    return p


# Example interactive usage:
if __name__ == "__main__":
    print("\n=== NBA Arbitrage Evaluator ===\n")

    home_ml = input("Home Moneyline (e.g. -120): ")
    away_ml = input("Away Moneyline (e.g. +140): ")

    # Example — you will swap with actual team rolling stats from DB
    home_feats = {
        "PTS_ROLL": 112,
        "REB_ROLL": 43,
        "AST_ROLL": 24,
        "STL_ROLL": 7,
        "BLK_ROLL": 4,
        "TOV_ROLL": 13,
        "FG_PCT_ROLL": .48,
        "FG3_PCT_ROLL": .37,
        "FT_PCT_ROLL": .78,
    }

    away_feats = {
        "PTS_ROLL": 109,
        "REB_ROLL": 41,
        "AST_ROLL": 23,
        "STL_ROLL": 6,
        "BLK_ROLL": 4,
        "TOV_ROLL": 14,
        "FG_PCT_ROLL": .47,
        "FG3_PCT_ROLL": .36,
        "FT_PCT_ROLL": .77,
    }

    # Predict
    prob = predict(home_feats, away_feats)

    print(f"\nModel Win Probability (Home): {prob:.3f}")

    # Evaluate for both sides
    print("\n--- HOME SIDE ---")
    print(evaluate_bet(prob, home_ml))

    print("\n--- AWAY SIDE ---")
    print(evaluate_bet(1 - prob, away_ml))
