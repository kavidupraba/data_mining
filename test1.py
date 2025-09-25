from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.graphics.tsaplots as F

df_d=pd.read_csv("./data/day.csv")
d_s=df_d.sample(n=3,replace=False,random_state=42)#n=3 (show 3 rows random) repalce=Fals the sample rows are unique random_state
#so I can generate same result again and again
print(d_s)

df_h=pd.read_csv("./data/hour.csv")
d_hs=df_h.sample(n=3,replace=False,random_state=42)
print(d_hs)

#converting date and time to datetime
df_d['dteday']=pd.to_datetime(df_d['dteday'])
df_h['dteday']=pd.to_datetime(df_h['dteday'])

#print(df_d.head())
#print(df_h.head())

#setting index to columns
df_d.set_index("dteday",inplace=True)
df_h.set_index("dteday",inplace=True)


print(df_d.head())
print(df_h.head())

#basic information
print(f"info: {df_d.info()}")
print(f"description {df_h.describe()}")

plt.figure(figsize=(12,5))
plt.plot(df_d.index,df_d['cnt'],label="Daily Rental")
plt.title("daily rental")
plt.xlabel("days")
plt.ylabel("rencount")
plt.show()

# time seriouse decompostion

#daily
result_day=seasonal_decompose(df_d["cnt"],model='additive',period=360 )#we ise addoctive model here most of the time sesonal change to the overall 
#rental is the same like 2011-2012 when summer there is close to 1500 increase of rental and winter this will decrease by close to 1500 
#each year it stayed almost the same additive method don't over astimate the growth example 
#additive model y(t)=trend(t)+seasonal(t)+residual(t)
#mutliplicative model y(t)=trent(t)*seasonal(t)*residual(t)
#observ-overall rental count
#trend-smooth growth or decrease
#seasonal-repeating pattern
#residual-noise
result_day.plot()
plt.suptitle("Daily rental decomposition",fontsize=16)
plt.show()

#agrigate by day and decompose by houre to check (peak hourse) is each peak reapeat daily example 8AM is a rush hourre is it 
# repeate each day with this method we can reduce row count and analize peak and low in rental count
hourly_sum=df_h.groupby([df_h.index,'hr'])['cnt'].sum().unstack()
print(hourly_sum)

#flatten sum accrose the hours to get total cnt for days
print(df_h.index)
daily_sum=df_h.groupby([df_h.index])['cnt'].sum()#we don't need to use unstack here because there is only two 1 column dteday is 
#index here 
daily_sum=pd.DataFrame(data=daily_sum)
print(daily_sum)

#now we can check if day.csv->df_d) 'cnt' and daily_sum that we derived from hour.csv->df_h->daily_sum 'cnt' is the same
result=df_d["cnt"]==daily_sum["cnt"]
result=np.array(result)
accuracy=np.mean(result)
print(f"accuracy: {accuracy*100} %")

#hourly agrigated rental decomposition 
result_h=seasonal_decompose(daily_sum,model='additive',period=7)#7day sycle-week
result_h.plot()
plt.suptitle("Hourly Aggregated Rentals Decomposition",fontsize=16)
plt.show()

#Autocorrilation

#Autocorrilation
plt.figure(figsize=(10,4))
F.plot_acf(df_d["cnt"],lags=50)
plt.title("Autocorrilation (Full) comparing each peak/low")#how was today sale compare to yesterday ,daybefore ,...n day before
plt.show()

#Partial Autocorilation
plt.figure(figsize=(10,4))
F.plot_pacf(df_d["cnt"],lags=50)
plt.title("Partial Autocorrilation comparing peaks and lows truly matters")#Is different between today and n day before is truly matters then show it
plt.show()