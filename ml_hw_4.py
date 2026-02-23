import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

fin_data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
fin_data.columns = [col[0] if isinstance(col, tuple) else col for col in fin_data.columns]
fin_data['Return_1d']=fin_data['Close'].pct_change()
fin_data['MA_5']=fin_data['Close'].rolling(5).mean()
fin_data['MA_10']=fin_data['Close'].rolling(10).mean()
fin_data['Volatility_5']=fin_data['Return_1d'].rolling(5).std()
fin_data['Volume_Change']=fin_data['Volume'].pct_change()

fin_data['Target']=(fin_data['Return_1d'].shift(-1)>0).astype(int)
fin_data.dropna(inplace=True)

fin_data.info()

X = fin_data[['Return_1d','MA_5',"MA_10",'Volatility_5','Volume_Change']].values
y=fin_data['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42,shuffle=True)

def cross_val(X,y,mod_class,mod_param,n_folds=5):
    np.random.seed(42)
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    fold_size = len(X) // n_folds
    scores = []
    scores_2 = []

    for fold in range(n_folds):
        start = fold*fold_size
        
        end=start+fold_size
        val_id = ind[start:end]

        train_id = np.concatenate((ind[:start],ind[end:]))
        X_train, y_train=X[train_id],y[train_id]
        X_val,y_val=X[val_id],y[val_id]

        if len(X_train) ==0 or len(X_val) ==0:
            continue

        scaler=StandardScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model=mod_class(**mod_param)
        model.fit(X_train_scaled,y_train)

        y_pred = model.predict(X_val_scaled)
        y_pred_2 = model.predict(X_train_scaled)
        acc=accuracy_score(y_val,y_pred)
        acc_2 = accuracy_score(y_train,y_pred_2)
        scores.append(acc)
        scores_2.append(acc_2)

        print(f"Fold{fold+1}: accuracy test ={acc:.4f} accuracy train: {acc_2:.4f}")

    return np.mean(scores)

C = [.01,.1,1,10]
penalties=['l1','l2']

best_score = 0
best_params = None

for c in C:
    for p in penalties:
        score=cross_val(
            X_train,y_train,
            mod_class=LogisticRegression,
            mod_param={"C":c,'penalty':p,'solver':'liblinear'},
            n_folds=5
        )
        print(f"C={c}, penalty={p}, CV score ={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = (c,p)

print("best parameters: ", best_params)
print("BEst mean CV accuracy", best_score)




