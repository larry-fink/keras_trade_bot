
"""
Created on Thu Nov  2 18:39:39 2017

"""

import numpy
import pandas
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import json
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
import urllib
import urllib.request
import random

###### defining functions:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back][0])
    return numpy.array(dataX), numpy.array(dataY)


# putting data into dataframes:
def filldf(dataframe,column,source):
    for i in range(len(dataframe)):
        dataframe[column][i] = source[i][column]
    return dataframe






# fix random seed for reproducibility
numpy.random.seed(13)

          


###### loading data

# main currency
#local_filename, headers = urllib.request.urlretrieve('https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1456189200&end=9999999999&period=7200')
urlData = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1456189200&end=9999999999&period=7200'
webURL = urllib.request.urlopen(urlData)
data = webURL.read()
encoding = webURL.info().get_content_charset('utf-8')
data = json.loads(data.decode(encoding))



# BTC (main correlator)

#local_filename, headers = urllib.request.urlretrieve('https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1456189200&end=9999999999&period=7200')

urlData = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1456189200&end=9999999999&period=7200'
webURL = urllib.request.urlopen(urlData)
databtc = webURL.read()
encoding = webURL.info().get_content_charset('utf-8')
databtc = json.loads(databtc.decode(encoding))




###### creating dataframes:

columns = ['close','weightedAverage','volume', 'open' ]
index = range(len(data))
df = pandas.DataFrame(index=index, columns=columns)


columns = ['close','weightedAverage','volume' ]
index = range(len(databtc))
df_btc = pandas.DataFrame(index=index, columns=columns)





###### Data preprocessing:

#main coin
df = filldf(df,'weightedAverage',data)
df = filldf(df,'volume',data)
df = filldf(df,'open',data)
df = filldf(df,'close',data)

print(df.head())
print(df.tail())

#btc
df_btc = filldf(df_btc,'weightedAverage',databtc)
df_btc = filldf(df_btc,'volume',databtc)
df_btc = filldf(df_btc,'close',databtc)

print(df_btc.head())
print(df_btc.tail())

#mergig dataframes
df_concat = pandas.concat([df.reset_index(drop=True),df_btc],axis=1, )
df_concat.columns = ['close','weightedAverage','volume', 'open','close_btc','weightedAverage_btc','volume_btc' ]

print(df_concat.head())

#df['weightedAverage'].plot()
#pyplot.show()

dataset = df_concat[['close','weightedAverage','volume', 'open','close_btc','weightedAverage_btc','volume_btc' ]].values


#Adding RSI:

delta = df['close'].diff()
delta = delta[1:]
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0
roll_up1 = pandas.stats.moments.ewma(up, 14)
roll_down1 = pandas.stats.moments.ewma(down.abs(), 14)
RS1 = roll_up1 / roll_down1
RSI1 = 100.0 - (100.0 / (1.0 + RS1))

RSI1 = numpy.array(RSI1)
RSI1 = RSI1.reshape(RSI1.shape[0],1)



roll_up1 = pandas.stats.moments.ewma(up, 7)
roll_down1 = pandas.stats.moments.ewma(down.abs(), 7)
RS2 = roll_up1 / roll_down1
RSI2 = 100.0 - (100.0 / (1.0 + RS2))

RSI2 = numpy.array(RSI2)
RSI2 = RSI2.reshape(RSI2.shape[0],1)


dataset = numpy.concatenate((dataset[1:dataset.shape[0],:],RSI1, RSI2),axis=1)

rolling5 = df['weightedAverage'].rolling(5).mean()
rolling5 = numpy.array(rolling5)
rolling5 = rolling5.reshape(rolling5.shape[0],1)


dataset = numpy.concatenate((dataset,rolling5[1:len(rolling5),]),axis=1)


#Adding BB:

BB_u = pandas.rolling_mean(df['close'], window=20) + 2* pandas.rolling_std(df['close'], 20, min_periods=20)
BB_d = pandas.rolling_mean(df['close'], window=20) - 2* pandas.rolling_std(df['close'], 20, min_periods=20)

BB_u = numpy.array(BB_u)
BB_u = BB_u.reshape(BB_u.shape[0],1)

BB_d = numpy.array(BB_d)
BB_d = BB_u.reshape(BB_d.shape[0],1)



dataset = numpy.concatenate((dataset[0:dataset.shape[0],:],BB_u[1:len(BB_u),:], BB_d[1:len(BB_d), :]),axis=1)

#excluding first 20 lines:
dataset = dataset[20:len(dataset),:]

spamset = dataset

# normalize the dataset
train_size = int(len(dataset[100:len(dataset),:]) * 0.95)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(dataset[100:train_size,:])
dataset = scaler.transform(dataset[100:len(dataset),:])


# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train, test = dataset[:train_size], dataset[train_size:len(dataset)]


#random noise
test= test + numpy.random.normal(0,0.005,test.shape)




# reshape into X=t and Y=t+1
look_back = 24
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)



# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 12))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 12))


###### Model building:

#optimizer:

myadam = keras.optimizers.Adam(lr=0.00001)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(30, input_shape=(look_back, 12), return_sequences = True, dropout=0.2))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(20))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=250, batch_size=200, verbose=2, shuffle=False)




###### Predictions and testing:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
#trainPredict = numpy.concatenate((trainPredict,dataset[:len(trainPredict),1:9]),axis=1)
#trainPredict = scaler.inverse_transform(trainPredict)[:,0]
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])


trainPredict = trainPredict * spamset[100: (100+train_size),0].max()#df['weightedAverage'].max()
trainY = trainY * spamset[100:train_size,0].max() #df['weightedAverage'].max()
testPredict = testPredict *  spamset[100:train_size,0].max()#df['weightedAverage'].max()
testY = testY * spamset[100:train_size,0].max()#df['weightedAverage'].max()





###### Results and plotting:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore)) #0.18 RMSE
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore)) #0.25 RMSE
print(mean_absolute_error(testY, testPredict[:,0]))  #0.184
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset[:,0])
trainPredictPlot[:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict[:,0]
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset[:,0])
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict[:,0]
# plot baseline and predictions
plt.plot(dataset[:,0]*  spamset[100:train_size,0].max() ) #df['weightedAverage'].max())
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



###### backtesting:

count = 100
treas = 0
portfolio = []
order_exec_prob = 0.9


for i in range(len(testY)-1):
        if (testPredict[i+1,0] > testY[i]) and (random.random()<order_exec_prob ):
                treas += (count/testY[i])*0.9975
                count = 0
        elif (testPredict[i+1,0] < testY[i]) and  treas > 0 and (random.random()<order_exec_prob ) :
                count = (treas * testY[i]) *0.9975
                treas = 0

        else:
                pass

        #print(i)
        #print(count)
        #print(treas)
        portfolio.append(count+treas*testY[i])
                

print("final results")
print(treas)  # 4.84
print(count)  # 1038
print(treas*testY[i])                
print(testY[i]/testY[0])
plt.plot(testY/testY[0]*100)
plt.plot(portfolio)
plt.show()




###### meta profitability:

meta_portfolio = []
order_exec_prob = 0.7

for j in range(10000):
    count = 100
    treas = 0
    portfolio = []
    
    
    
    for i in range(len(testY)-1):
            if (testPredict[i+1,0] > testY[i]) and (random.random()<order_exec_prob ):
                    treas += (count/testY[i])*0.9975
                    count = 0
            elif (testPredict[i+1,0] < testY[i]) and  treas > 0 and (random.random()<order_exec_prob ) :
                    count = (treas * testY[i]) *0.9975
                    treas = 0
    
            else:
                    pass
    
            #print(i)
            #print(count)
            #print(treas)
            portfolio.append(count+treas*testY[i])
    
    meta_portfolio.append(count+treas*testY[i])
    #print(i)

plt.hist(meta_portfolio)
numpy.mean(meta_portfolio)
numpy.median(meta_portfolio)

'''
###### suboptimal backtesting:
count = 100
treas = 0
portfolio = []

for i in range(len(testY)-1):
        if testPredict[i+1,0] > testY[i]*1.001:
                treas += (count/(testY[i]*1.0005))*0.9975
                count = 0
        elif (testPredict[i+1,0] < testY[i]*0.999) and  treas > 0 :
                count = (treas * (testY[i]*0.9985)) *0.9975
                treas = 0

        else:
                pass

        #print(i)
        #print(treas)
        #print(count)
        portfolio.append(count+treas*testY[i])
        
                

print("final results")
print(treas)  # 3.949
print(count)  #  986.45930372749001
print(treas*testY[i])
print(testY[i]/testY[0])
plt.plot(testY/testY[0]*100)
plt.plot(portfolio)
plt.show()






###### suboptimal backtesting:
count = 100
treas = 0
portfolio = []

for i in range(len(testY)-1):
        if testPredict[i+1,0] > testY[i]*1.002:
                treas += (count*0.7/(testY[i]*1.002))*0.9975
                count -= count*0.7
        elif (testPredict[i+1,0] < testY[i]*0.998) and  treas > 0 :
                count += (treas * (testY[i]*0.998)) *0.9975
                treas -= treas

        else:
                pass

        #print(i)
        #print(treas)
        #print(count)
        portfolio.append(count+treas*testY[i])
        
                

print("final results")
#print(treas)  # 3.949
#print(count)  #  986.45930372749001
#print(treas*testY[i])
print(testY[i]/testY[0])
print(count+treas*testY[i])
#print(portfolio)
plt.plot(testY/testY[0]*100)
plt.plot(portfolio)
plt.show()
'''



