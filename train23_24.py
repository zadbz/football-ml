# train_23_24_predict_25.py
import os, requests, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

API="https://api.football-data.org/v4"; HEAD={"X-Auth-Token":os.getenv("FOOTBALL_DATA_TOKEN")}
TOP5=["PL","PD","SA","BL1","FL1"]              # EPL, LaLiga, Serie A, Bundesliga, Ligue 1
SEASONS_TRAIN=[2023,2024]                      # train on 23-24 & 24-25
TARGET_SEASON=2025                             # predict 25-26
HFA=60; K=20

def fetch_matches(comp, season):
    r=requests.get(f"{API}/competitions/{comp}/matches",params={"season":season},headers=HEAD,timeout=30)
    if r.status_code==403: print(f"[WARN] 403 {comp} {season} -> skipping"); return pd.DataFrame()
    r.raise_for_status()
    rows=[]
    for m in r.json().get("matches",[]):
        if m["status"] not in ("FINISHED","TIMED","SCHEDULED"): continue
        rows.append({"comp":comp,"season":season,"utc":m["utcDate"],"home":m["homeTeam"]["shortName"],
                     "away":m["awayTeam"]["shortName"],"hid":m["homeTeam"]["id"],"aid":m["awayTeam"]["id"],
                     "status":m["status"],"hg":m["score"]["fullTime"]["home"],"ag":m["score"]["fullTime"]["away"]})
    df=pd.DataFrame(rows); 
    if df.empty: return df
    df["date"]=pd.to_datetime(df["utc"]); 
    return df.sort_values("date").reset_index(drop=True)

def res(hg,ag):
    if pd.isna(hg) or pd.isna(ag): return None
    return 0 if hg>ag else (1 if hg==ag else 2)

def upd(eh,ea,r):
    Eh=1/(1+10**(((ea)-(eh+HFA))/400)); Sh,Sa=(1,0) if r==0 else ((0.5,0.5) if r==1 else (0,1))
    return eh+K*(Sh-Eh), ea+K*(Sa-(1-Eh))

def build_feats(df, elo=None):
    elo=elo or {}; feats=[]
    for _,x in df.iterrows():
        eh,ea=elo.get(x.hid,1500.0),elo.get(x.aid,1500.0)
        feats.append({"date":x.date,"comp":x.comp,"season":x.season,"home":x.home,"away":x.away,
                      "hid":x.hid,"aid":x.aid,"elo_h":eh,"elo_a":ea,"elo_diff":(eh+HFA)-ea,
                      "status":x.status,"hg":x.hg,"ag":x.ag})
        r=res(x.hg,x.ag)
        if r is not None: elo[x.hid],elo[x.aid]=upd(eh,ea,r)
    X=pd.DataFrame(feats); X["y"]=[res(hg,ag) for hg,ag in zip(X.hg,X.ag)]
    return X, elo

def train_on_seasons():
    frames=[]; 
    for comp in TOP5:
        for s in SEASONS_TRAIN:
            d=fetch_matches(comp,s)
            if not d.empty: frames.append(d)
    if not frames: raise SystemExit("No training data pulled (plan likely blocks history).")
    allm=pd.concat(frames).sort_values("date").reset_index(drop=True)
    X,elo=build_feats(allm)
    train=X[X.y.notna()]
    model=LogisticRegression(max_iter=1000)  # multinomial by default in new sklearn
    model.fit(train[["elo_diff"]],train["y"].astype(int))
    split=int(len(train)*0.8); va=train.iloc[split:]; pr=model.predict_proba(va[["elo_diff"]])
    metrics={"log_loss":float(log_loss(va["y"],pr)),
             "accuracy":float(accuracy_score(va["y"],pr.argmax(1))),
             "n_train":int(len(train)),"n_val":int(len(va))}
    return model, elo, metrics

def predict_target(model, seed_elo):
    frames=[]; 
    for comp in TOP5:
        d=fetch_matches(comp, TARGET_SEASON)
        if not d.empty: frames.append(d)
    if not frames: raise SystemExit("No target-season data (check plan or season code).")
    tgt=pd.concat(frames).sort_values("date").reset_index(drop=True)
    X, _ = build_feats(tgt, elo=seed_elo.copy())  # use Elo carried from training
    P=pd.DataFrame(model.predict_proba(X[["elo_diff"]]),columns=["p_home","p_draw","p_away"])
    out=pd.concat([X.reset_index(drop=True),P],axis=1)
    return out

def main():
    if not HEAD["X-Auth-Token"]: raise SystemExit("FOOTBALL_DATA_TOKEN not set.")
    model, elo, metr = train_on_seasons()
    print("Train metrics:", metr)
    preds = predict_target(model, elo)
    preds.to_csv("predictions_target_2025_all.csv",index=False)
    played = preds[preds.y.notna()].copy()
    if not played.empty:
        pr=played[["p_home","p_draw","p_away"]].to_numpy()
        yy=played["y"].astype(int).to_numpy()
        print("2025-to-date:",
              {"log_loss":float(log_loss(yy,pr)),"accuracy":float((pr.argmax(1)==yy).mean()),
               "n":int(len(played))})
        played.to_csv("predictions_target_2025_played.csv",index=False)
    future = preds[preds.status.isin(["TIMED","SCHEDULED"])].sort_values(["date","comp"])
    future.to_csv("predictions_target_2025_future.csv",index=False)
    print("Wrote: predictions_target_2025_all.csv / _played.csv / _future.csv")

if __name__=="__main__": main()
