import os, requests, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

API="https://api.football-data.org/v4"; TOKEN=os.getenv("FOOTBALL_DATA_TOKEN")
HEAD={"X-Auth-Token":TOKEN}

def fetch_matches(comp="PL", season=2024):
    r=requests.get(f"{API}/competitions/{comp}/matches?season={season}",headers=HEAD,timeout=30); r.raise_for_status()
    rows=[]
    for m in r.json().get("matches",[]):
        if m["status"] not in ("FINISHED","TIMED","SCHEDULED"): continue
        rows.append({"utcDate":m["utcDate"],"home":m["homeTeam"]["shortName"],"away":m["awayTeam"]["shortName"],
                     "homeId":m["homeTeam"]["id"],"awayId":m["awayTeam"]["id"],"status":m["status"],
                     "homeGoals":m["score"]["fullTime"]["home"],"awayGoals":m["score"]["fullTime"]["away"]})
    df=pd.DataFrame(rows); df["date"]=pd.to_datetime(df["utcDate"]); return df.sort_values("date").reset_index(drop=True)

def result(hg,ag):
    if pd.isna(hg) or pd.isna(ag): return None
    return 0 if hg>ag else (1 if hg==ag else 2)  # 0=H,1=D,2=A

def update_elo(eh,ea,res,k=20,HFA=60):
    Eh=1/(1+10**(((ea)-(eh+HFA))/400)); Ea=1-Eh
    Sh,Sa=(1,0) if res==0 else ((0.5,0.5) if res==1 else (0,1))
    return eh+k*(Sh-Eh), ea+k*(Sa-Ea)

def build_dataset(df):
    elo={}; feats=[]
    for _,r in df.iterrows():
        h,a=r.homeId,r.awayId; eh,ea=elo.get(h,1500.0),elo.get(a,1500.0)
        feats.append({"date":r.date,"homeId":h,"awayId":a,"elo_h":eh,"elo_a":ea,"elo_diff":(eh+60)-ea,
                      "status":r.status,"home":r.home,"away":r.away,"homeGoals":r.homeGoals,"awayGoals":r.awayGoals})
        res=result(r.homeGoals,r.awayGoals)
        if res is not None: elo[h],elo[a]=update_elo(eh,ea,res)
    X=pd.DataFrame(feats); X["y"]=[result(hg,ag) for hg,ag in zip(X.homeGoals,X.awayGoals)]; return X

def train_logit(X):
    train=X[X.y.notna()]; model=LogisticRegression(multi_class="multinomial",max_iter=1000)
    model.fit(train[["elo_diff"]],train["y"].astype(int))
    split=int(len(train)*0.8); va=train.iloc[split:]; pr=model.predict_proba(va[["elo_diff"]])
    return model, {"log_loss":log_loss(va["y"],pr), "accuracy":accuracy_score(va["y"],pr.argmax(1))}

def predict(model,X):
    P=pd.DataFrame(model.predict_proba(X[["elo_diff"]]),columns=["p_home","p_draw","p_away"])
    out=pd.concat([X.reset_index(drop=True),P],axis=1)
    for c in ("p_home","p_draw","p_away"): out[f"odds_{c[2:]}"]=1/np.clip(out[c],1e-6,1)
    return out

def make_parlay(pred_df,max_legs=3,min_prob=0.6):
    legs=[]
    for _,r in pred_df.sort_values(["date","p_home","p_draw","p_away"],ascending=[True,False,False,False]).iterrows():
        opt=np.argmax([r.p_home,r.p_draw,r.p_away]); prob=[r.p_home,r.p_draw,r.p_away][opt]
        if prob<min_prob or len(legs)>=max_legs: continue
        legs.append({"date":r.date.date(),"home":r.home,"away":r.away,"pick":["H","D","A"][opt],"prob":prob,"decimal_odds":1/max(prob,1e-6)})
    if not legs: return pd.DataFrame(),None,None
    return pd.DataFrame(legs), float(np.prod([x["decimal_odds"] for x in legs])), float(np.prod([x["prob"] for x in legs]))

def main():
    df=fetch_matches("PL",season=2024); X=build_dataset(df); model,metrics=train_logit(X); print("Metrics:",metrics)
    preds=predict(model,X); future=preds[preds.status.isin(["TIMED","SCHEDULED"])].sort_values("date")
    legs,odds,prob=make_parlay(future)
    preds.to_csv("predictions_all.csv",index=False); future.to_csv("predictions_future.csv",index=False)
    if legs is not None and len(legs): legs.to_csv("parlay_suggestion.csv",index=False); print("Parlay:",legs.to_dict(orient="records"),"| odds:",round(odds,2),"| prob:",round(prob,3))
    else: print("No parlay found.")
if __name__=="__main__": main()
