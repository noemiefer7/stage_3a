import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


df = pd.read_excel('WL_Ladji.xlsx', header = 0, parse_dates = True)
time_list = ['']*len(df.axes[0])

for i in range( len(df.axes[0])):
   
    time_list[i]+=(str(df.iloc[i,0]) + '-' + str(df.iloc[i,1]) + '-' + str(df.iloc[i,2]) + ' ' + str(df.iloc[i,3]) + ':' + str(df.iloc[i,4]) + ':00')
   
            
fig = go.Figure([go.Scatter(x = time_list, y = df['Water high'])])
fig.show()

# plt.figure()
# plt.plot(water_height