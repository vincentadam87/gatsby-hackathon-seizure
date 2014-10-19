import pandas as pd
data = pd.read_csv('out.csv',header=0)
y_s = data['seizure']
y_e = data['early']
data2 = data
data2['early'] = y_e*y_s
data2.to_csv('out_hack.csv',sep=',')
