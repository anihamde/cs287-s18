import pandas as pd
df_set = []
for file in sys.argv[1:]:
    df_set.append( pd.read_csv(file,index_col=0) )

pred = pd.concat(df_set,axis=1).mode(axis=1)
pred.columns = ["Cat"]
pred.to_csv("../data/cnn_ensemble1_predictions.csv",header=True)
