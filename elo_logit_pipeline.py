import os, requests, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

API="https://api.football-data.org/v4"; HEAD={"X-Auth-Token":os.getenv("FOOTBALL_DATA_TOKEN")}
TOP5=["PL","PD","SA","BL1","FL1"]  # EPL, LaLiga, Serie A, Bundesliga, Ligue 1
SEASONS=list(range(pd.Timestamp.utcnow().year-1, pd.Timestamp.utcnow().year-6, -1))  # last 5 seasons by start-year guess

def fetch_matches(comp, season):
    r=requests.get(f"{API}/competitions/{comp}/matches",params={"season":season},headers=HEAD,timeout=30); r.raise_for_status()
    rows=[]
    for m in r.json().get("matches",[]):
        if m["status"] not in ("FINISHED","TIMED","SCHEDULED"): continue
        rows.append({"comp":comp,"season":season,"utcDate":m["utcDate"],"home":m["homeTeam"]["shortName"],"away":m["awayTeam"]["shortName"],
                     "homeId":m["homeTeam"]["id"],"awayId":m["awayTeam"]["id"],"status":m["status"],
                     "homeGoals":m["score"]["fullTime"]["home"],"awayGoals":m["score"]["fullTime"]["away"]})
    df=pd.DataFrame(rows); 
    if len(df)==0: return df
    df["date"]=pd.to_datetime(df["utcDate"]); 
    return df.sort_values("date").reset_index(drop=True)

def result(hg,ag):
    if pd.isna(hg) or pd.isna(ag): return None
    return 0 if hg>ag else (1 if hg==ag else 2)

def update_elo(eh,ea,res,k=20,HFA=60):
    Eh=1/(1+10**(((ea)-(eh+HFA))/400)); Sh,Sa=(1,0) if res==0 else ((0.5,0.5) if res==1 else (0,1))
    return eh+k*(Sh-Eh), ea+k*(Sa-(1-Eh))

def build_dataset(all_matches):
    elo, feats = {}, []
    for _,r in all_matches.iterrows():
        h,a=r.homeId,r.awayId; eh,ea=elo.get(h,1500.0),elo.get(a,1500.0)
        feats.append({"date":r.date,"comp":r.comp,"season":r.season,"home":r.home,"away":r.away,"homeId":h,"awayId":a,
                      "elo_h":eh,"elo_a":ea,"elo_diff":(eh+60)-ea,"status":r.status,"homeGoals":r.homeGoals,"awayGoals":r.awayGoals})
        res=result(r.homeGoals,r.awayGoals)
        if res is not None: elo[h],elo[a]=update_elo(eh,ea,res)
    X=pd.DataFrame(feats); X["y"]=[result(hg,ag) for hg,ag in zip(X.homeGoals,X.awayGoals)]; return X

def train_logit(X):
    train=X[X.y.notna()].copy()
    model=LogisticRegression(multi_class="multinomial",max_iter=1000)
    model.fit(train[["elo_diff"]],train["y"].astype(int))
    split=int(len(train)*0.8); va=train.iloc[split:]; pr=model.predict_proba(va[["elo_diff"]])
    return model, {"log_loss":log_loss(va["y"],pr), "accuracy":accuracy_score(va["y"],pr.argmax(1)), "n_train":len(train),"n_val":len(va)}

def predict(model,X):
    P=pd.DataFrame(model.predict_proba(X[["elo_diff"]]),columns=["p_home","p_draw","p_away"])
    out=pd.concat([X.reset_index(drop=True),P],axis=1)
    for c in ("p_home","p_draw","p_away"): out[f"odds_{c[2:]}"]=1/np.clip(out[c],1e-6,1)
    return out

def make_parlay(pred_df,max_legs=3,min_prob=0.6):
    legs=[]
    cand=pred_df[pred_df.status.isin(["TIMED","SCHEDULED"])].copy()
    cand=cand.sort_values(["date"]).copy()
    for _,r in cand.itertuples(index=False):
        # not used because of namedtuple; fallback below
        pass
    for _,r in cand.iterrows():
        opt=np.argmax([r.p_home,r.p_draw,r.p_away]); prob=[r.p_home,r.p_draw,r.p_away][opt]
        if prob<min_prob or any(r.home in (x["home"],x["away"]) or r.away in (x["home"],x["away"]) for x in legs): continue
        legs.append({"date":r.date.date(),"comp":r.comp,"home":r.home,"away":r.away,"pick":["H","D","A"][opt],"prob":prob,"decimal_odds":1/prob})
        if len(legs)>=max_legs: break
    if not legs: return pd.DataFrame(),None,None
    return pd.DataFrame(legs), float(np.prod([x["decimal_odds"] for x in legs])), float(np.prod([x["prob"] for x in legs]))

def main():
    frames=[]
    for comp in TOP5:
        for season in SEASONS:
            df=fetch_matches(comp,season); 
            if len(df): frames.append(df)
    if not frames: raise SystemExit("No data pulled. Check token or API plan.")
    all_matches=pd.concat(frames).sort_values("date").reset_index(drop=True)
    X=build_dataset(all_matches); model,metrics=train_logit(X); print("Metrics:",metrics)
    preds=predict(model,X); preds.to_csv("predictions_all.csv",index=False)
    future=preds[preds.status.isin(["TIMED","SCHEDULED"])].sort_values(["date","comp"])
    future.to_csv("predictions_future.csv",index=False)
    legs,odds,prob=make_parlay(future)
    if legs is not None and len(legs): legs.to_csv("parlay_suggestion.csv",index=False); print("Parlay:",legs.to_dict(orient="records"),"| odds:",round(odds,2),"| prob:",round(prob,3))
    else: print("No parlay found.")
if __name__=="__main__": main()
