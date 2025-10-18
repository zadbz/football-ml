import os, requests, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

API = "https://api.football-data.org/v4"
TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
HEAD = {"X-Auth-Token": TOKEN}
TOP5 = ["PL", "PD", "SA", "BL1", "FL1"]  # EPL, LaLiga, Serie A, Bundesliga, Ligue 1
CURRENT = 2024  # change if needed

def fetch_matches(comp, season=CURRENT):
    r = requests.get(f"{API}/competitions/{comp}/matches",
                     params={"season": season}, headers=HEAD, timeout=30)
    if r.status_code == 403:
        print(f"[WARN] 403 {comp} {season} (plan restriction) -> skipping")
        return pd.DataFrame()
    r.raise_for_status()
    rows = []
    for m in r.json().get("matches", []):
        if m["status"] not in ("FINISHED", "TIMED", "SCHEDULED"): continue
        rows.append({
            "comp": comp, "season": season, "utcDate": m["utcDate"],
            "home": m["homeTeam"]["shortName"], "away": m["awayTeam"]["shortName"],
            "homeId": m["homeTeam"]["id"], "awayId": m["awayTeam"]["id"],
            "status": m["status"],
            "homeGoals": m["score"]["fullTime"]["home"],
            "awayGoals": m["score"]["fullTime"]["away"]
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["utcDate"])
    return df.sort_values("date").reset_index(drop=True)

def result(hg, ag):
    if pd.isna(hg) or pd.isna(ag): return None
    return 0 if hg > ag else (1 if hg == ag else 2)

def update_elo(eh, ea, res, k=20, HFA=60):
    Eh = 1/(1 + 10**(((ea) - (eh + HFA))/400))
    Sh, Sa = (1,0) if res==0 else ((0.5,0.5) if res==1 else (0,1))
    return eh + k*(Sh - Eh), ea + k*(Sa - (1 - Eh))

def build_dataset(all_matches):
    elo, feats = {}, []
    for _, r in all_matches.iterrows():
        eh, ea = elo.get(r.homeId,1500.0), elo.get(r.awayId,1500.0)
        feats.append({
            "date": r.date, "comp": r.comp, "season": r.season,
            "home": r.home, "away": r.away, "homeId": r.homeId, "awayId": r.awayId,
            "elo_h": eh, "elo_a": ea, "elo_diff": (eh + 60) - ea,
            "status": r.status, "homeGoals": r.homeGoals, "awayGoals": r.awayGoals
        })
        res = result(r.homeGoals, r.awayGoals)
        if res is not None:
            elo[r.homeId], elo[r.awayId] = update_elo(eh, ea, res)
    X = pd.DataFrame(feats)
    X["y"] = [result(hg, ag) for hg, ag in zip(X.homeGoals, X.awayGoals)]
    return X

def train_logit(X):
    train = X[X.y.notna()]
    model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    model.fit(train[["elo_diff"]], train["y"].astype(int))
    split = int(len(train)*0.8)
    va = train.iloc[split:]
    pr = model.predict_proba(va[["elo_diff"]])
    metrics = {"log_loss": log_loss(va["y"], pr),
               "accuracy": accuracy_score(va["y"], pr.argmax(1)),
               "n_train": len(train), "n_val": len(va)}
    return model, metrics

def predict(model, X):
    P = pd.DataFrame(model.predict_proba(X[["elo_diff"]]), columns=["p_home","p_draw","p_away"])
    out = pd.concat([X.reset_index(drop=True), P], axis=1)
    for c in ("p_home","p_draw","p_away"):
        out[f"odds_{c[2:]}"] = 1/np.clip(out[c], 1e-6, 1)
    return out

def main():
    if not TOKEN: raise SystemExit("FOOTBALL_DATA_TOKEN not set.")
    frames = []
    for comp in TOP5:
        df = fetch_matches(comp, CURRENT)
        if not df.empty: frames.append(df)
    if not frames: raise SystemExit("No data pulled (possibly plan-limited).")
    all_matches = pd.concat(frames).sort_values("date").reset_index(drop=True)
    X = build_dataset(all_matches)
    model, metrics = train_logit(X)
    print("Metrics:", metrics)
    preds = predict(model, X)
    preds.to_csv("predictions_all.csv", index=False)
    future = preds[preds.status.isin(["TIMED","SCHEDULED"])].sort_values(["date","comp"])
    future.to_csv("predictions_future.csv", index=False)
    print("Wrote predictions_all.csv & predictions_future.csv")

if __name__ == "__main__": main()
