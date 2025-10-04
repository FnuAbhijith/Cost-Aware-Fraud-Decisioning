import pandas as pd, numpy as np
rng = np.random.default_rng(42); n=3000
df = pd.DataFrame({
 "amount": rng.gamma(2., 30., n),
 "device_risk": rng.uniform(0,1,n),
 "geo_distance_km": rng.exponential(50,n),
 "bin_risk": rng.beta(2,8,n),
 "country": rng.choice(["US","GB","IN","DE"], n, p=[.6,.15,.2,.05]),
})
p = 1/(1+np.exp(-(0.02*df["amount"]+3*df["device_risk"]+0.01*df["geo_distance_km"]+2*df["bin_risk"]-2.5)))
df["is_fraud"] = (rng.uniform(0,1,n) < p).astype(int)
df.to_csv("data/transactions.csv", index=False)
print("Wrote data/transactions.csv", df.shape)
