from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.graphics.tsaplots as F
from statsmodels.tsa.stattools import adfuller

df_d=pd.read_csv("/home/jack/data_mining/PROJECT/data/day.csv")
d_s=df_d.sample(n=3,replace=False,random_state=42)#n=3 (show 3 rows random) repalce=Fals the sample rows are unique random_state
#so I can generate same result again and again
print(d_s)

df_h=pd.read_csv("/home/jack/data_mining/PROJECT/data/hour.csv")
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


print(f"duplicate values for date: {df_d.duplicated().sum()}")
print(f"duplicate values for houre: {df_h.duplicated().sum()}")


"""Encoding cathegorical data with one hot encoding """

encoded_h = pd.get_dummies(df_h, columns=['yr','hr','season','mnth','weekday','weathersit'
])
encoded_d = pd.get_dummies(df_d, columns=['yr','season','mnth','weekday','weathersit'
])
print(encoded_h)


"""
Think of result[0] (ADF stat) as your exam score
and value (critical value) as the pass mark at each level.

If your score (ADF stat) is lower (more negative) than the pass mark (critical value),
you pass the test → the data is stationary.
"""
#check if time series is stationary or not if it fail the test add fixed time saries as "name"+dif_log
def adf_test(series, feature_name):
    result = adfuller(series[feature_name])
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")
        series[f"{feature_name}+dif_log"]=np.log(series[feature_name].diff())
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
        if result[0] < value:
            print(f"At {key} level, series is stationary")
    return series

           
        

# plt.figure(figsize=(12,5))
# plt.plot(df_d.index,df_d['cnt'],label="Daily Rental")
# plt.title("daily rental")
# plt.xlabel("days")
# plt.ylabel("rencount")
# plt.show() 

# # time seriouse decompostion

# #daily
# result_day=seasonal_decompose(df_d["cnt"],model='additive',period=360 )#we ise addoctive model here most of the time sesonal change to the overall 
# #rental is the same like 2011-2012 when summer there is close to 1500 increase of rental and winter this will decrease by close to 1500 
# #each year it stayed almost the same additive method don't over astimate the growth example 
# #additive model y(t)=trend(t)+seasonal(t)+residual(t)
# #mutliplicative model y(t)=trent(t)*seasonal(t)*residual(t)
# #observ-overall rental count
# #trend-smooth growth or decrease
# #seasonal-repeating pattern
# #residual-noise
# result_day.plot()
# plt.suptitle("Daily rental decomposition",fontsize=16)
# plt.show()

# #agrigate by day and decompose by houre to check (peak hourse) is each peak reapeat daily example 8AM is a rush hourre is it 
# # repeate each day with this method we can reduce row count and analize peak and low in rental count
# hourly_sum=df_h.groupby([df_h.index,'hr'])['cnt'].sum().unstack()
# print(hourly_sum)

# #flatten sum accrose the hours to get total cnt for days
# print(df_h.index)
# daily_sum=df_h.groupby([df_h.index])['cnt'].sum()#we don't need to use unstack here because there is only two 1 column dteday is 
# #index here 
# daily_sum=pd.DataFrame(data=daily_sum)
# print(daily_sum)

# #now we can check if day.csv->df_d) 'cnt' and daily_sum that we derived from hour.csv->df_h->daily_sum 'cnt' is the same
# result=df_d["cnt"]==daily_sum["cnt"]
# result=np.array(result)
# accuracy=np.mean(result)
# print(f"accuracy: {accuracy*100} %")

# #hourly agrigated rental decomposition 
# result_h=seasonal_decompose(daily_sum,model='additive',period=7)#7day sycle-week
# result_h.plot()
# plt.suptitle("Hourly Aggregated Rentals Decomposition",fontsize=16)
# plt.show()

# #Autocorrilation

# #Autocorrilation
# plt.figure(figsize=(10,4))
# F.plot_acf(df_d["cnt"],lags=50)
# plt.title("Autocorrilation (Full) comparing each peak/low")#how was today sale compare to yesterday ,daybefore ,...n day before
# plt.show()

# #Partial Autocorilation
# plt.figure(figsize=(10,4))
# F.plot_pacf(df_d["cnt"],lags=50)
# plt.title("Partial Autocorrilation comparing peaks and lows truly matters")#Is different between today and n day before is truly matters then show it
# plt.show()

# #seasonal pattern

# #Rental by month
# df_d['month']=df_d.index.month
# df_d.groupby(['month'])['cnt'].mean().plot(kind='bar',figsize=(10,5),title="Average rental over montsh")
# plt.show()

# #Rental by weekdays
# df_d['weekday']=df_d.index.weekday
# df_d.groupby('weekday')['cnt'].mean().plot(kind='bar',figsize=(10,5),title="average over weekdays")
# plt.show()

# #Rental by season
# sns.boxenplot(x='season',y='cnt',data=df_d.reset_index())
# plt.title("Seasonal average")
# plt.show()

# #Retanl by hour
# df_h.groupby('hr')['cnt'].mean().plot(kind='bar',figsize=(10,5),title="Average rental over hour")
# plt.show()

# # ===========================
# # 7. External Factors
# # ===========================
# # Scatter plots with weather variables
# fig, axes = plt.subplots(1, 3, figsize=(18,5))
# sns.scatterplot(x='temp', y='cnt', data=df_d, ax=axes[0])
# sns.scatterplot(x='hum', y='cnt', data=df_d, ax=axes[1])
# sns.scatterplot(x='windspeed', y='cnt', data=df_d, ax=axes[2])
# axes[0].set_title("Rentals vs Temp")
# axes[1].set_title("Rentals vs Humidity")
# axes[2].set_title("Rentals vs Windspeed")
# plt.show()

# # Correlation heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(df_d[['temp','atemp','hum','windspeed','cnt']].corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

# # Rentals on holidays vs working days
# sns.boxplot(x='holiday', y='cnt', data=df_d.reset_index())
# plt.title("Holiday vs Non-Holiday Rentals")
# plt.show()

# sns.boxplot(x='workingday', y='cnt', data=df_d.reset_index())
# plt.title("Working Day vs Non-Working Day Rentals")
# plt.show()
