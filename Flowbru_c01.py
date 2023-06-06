# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:22:35 2018

@author: Solomon
"""

import numpy as np
import pandas as pd

from FCS import fcs_functions as fcs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os
from pprint import pprint


## font and color for ploting
font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 14}
color = [(174, 199, 232), (214, 39, 40)]  

for i in range(len(color)):    
    r, g, b = color[i]    
    color[i] = (r / 255., g / 255., b / 255.)      
         

## Path of the gauges data (raingauges, surface and wastewater flow guages)
##path to the csv data file
inputfile1 = r'C:/Users/carsk/OneDrive - KU Leuven/Thesis/Revised_RESULTS/Hyperpar_calculation/FCS_input_files/9modeledStns.csv'


##path to the csv file with station information
path_sw = r'C:/Users/carsk/OneDrive - KU Leuven/Thesis/Revised_RESULTS/Hyperpar_calculation/FCS_input_files/wastewatercollectorslocation_new.csv'

## dataframe from the CVS data file
df2 = pd.read_csv(inputfile1, sep=',', decimal='.', index_col=[0], parse_dates=True)
df2 = df2.sort_index(axis=1)
df2 = df2.astype(np.float32)
# Read data about the guaging stations (ID, name, and x, y coordinates)
path_rain = r'C:/Users/carsk/OneDrive - KU Leuven/Thesis/Revised_RESULTS/Hyperpar_calculation/FCS_input_files/Rainfall_guageing_stations.csv'
Rainguages = fcs.station_data(path_rain)
SWguages = fcs.station_data(path_sw)

# Lis of stations to be modelled
#flow =  ['C01', 'C02', 'C11', 'U05', 'U06', 'U09', 'U11', 'U17', 'U19'] 

flow =  ['C02' ]

lag_times = {#'C01': 12,
             'C02': 15,
              # 'C11': 9,
              # 'U05': 15,
              # 'U06': 15,
              # 'U09': 15,
              # 'U11': 9,
              # 'U17': 18,
              # 'U19': 18}
              }

##Create output folder

outfolder=r'C:/Users/carsk/OneDrive - KU Leuven/Thesis/Revised_RESULTS/Hyperpar_calculation/Modelling_result'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

## Create list of Rainguages IDs
rain = []
for rg in Rainguages:
    rain.append(rg.id)
    
## Creat list of Surfacwwater flow uages IDs
flow_st = []
for st in SWguages:
    flow_st.append(st.id)


names = []
rsquared = []
WL = ['U06','C02','U09', 'C081','C082', 'C981','C982'] ## list of station of Water level
n = 5 ## number of RF stations closest to the flow station considered

## Create dataframe of the rainfall of the n number of closest stations
df = pd.DataFrame()
for f in flow:
    print('station', f)
    
    if (f == st.id for st in SWguages):
        df_list = fcs.closest_stations(Rainguages, SWguages, f,n)
        #print('the flow station and the 10 closest RF stations are ' , df_list)
        df_list = [x for x in df_list if x  in df2.columns.values]   #df2 is the dataframe containing the data read from the CSV
        df = df2[df_list].copy() # the final dataframe for the DDM
        names.append(f)
    else:
          print('No {0} in flow stations list'.format(f))  
    
    # ## if there are missing values, fill them with the mean of the series
    # for col in df.columns.values:
    #     df[col].fillna(df[col].mean(), inplace=True)
    #     #df[col][df[col]<0] = df[col].mean()
        
    # abstract the data of the required length (here cosider only from 2016 to 2018)
    df = df.loc['2016-01-01':'2018-12-31']
   
    
 #%%   
    # get list of Rainfall stations associated with the flow station
    rfst =  list(df.columns.values)
    del rfst[0]  # the first in the list id the flow station
    nshift = 24  ## the lag time in 5 minutes interval (2hr = 24, 1hr = 12) 
    rg_names = [] ## list of correlated rf gauging stations name
    
    corrST = []
    maxcorr = []
    indmaxcorr = [] ##index of the correlated RF station  
    corrRG = [] 
    
    for rg in rfst:
        #df[rg] = np.around(df[rg].rolling(nshift).apply(np.sum), decimals=2)
        rg_lag = rg+'_rs'+str(lag_times[f])
        # df[rg_lag] = df[rg].rolling(nshift).sum()
        df[rg_lag] = np.around(df[rg].rolling(lag_times[f]).apply(np.sum, raw=True), decimals=2)
        corr = df[f].corr(df[rg_lag])
        corrRG.append(corr)
        rg_names.append(rg_lag)
        sw2rg_corr = dict(zip(rg_names, corrRG))
        
    sorted_corr_list = sorted(sw2rg_corr, key=sw2rg_corr.get, reverse=True)

    # sorted_rg = [s.replace('_lag', '') for s in sorted_corr_list]
    # sorted_rg=sorted_corr_list
    nrfs2c = 5  ## number of RF stations to consider    
    # df_rg = df[sorted_rg[:nrfs2c]]
    df_rg = df[sorted_corr_list[:nrfs2c]]
    if(df_rg.isnull().values.any()):
        print('there is nan in the rainfall data')
    
#%%
     
    # lag the flow series by nshift ( the duration of the forecast time)   
    df_lag = pd.DataFrame()
    for i in range(10):
        #df[rg] = np.around(df[rg].rolling(nshift).apply(np.sum), decimals=2)
        lag = 'lag_'+str(i)
        name = f+'_lag_'+str(nshift+i)
        df_lag[name] = df[f].shift(nshift+i)

    
    df4 = pd.concat([df[f], df_lag], axis=1)
    if(df4.isnull().values.any()):
        print('there is nan in df[f]')

    if(not f in WL):
        df4 = df4[df4>=0]

        
    # add the 5 most correlated rainfall stations to the flow data frame
    df4 = pd.concat([df4, df_rg], axis=1)
    del df
    del df2
    del df_rg
    del df_lag
    import gc
    gc.collect()  # Manual garbage collection    
    if(df4.isnull().values.any()):
        print('there is nan in df4')
        
    # drop the nans
    df4 = df4.dropna()
    df4 = df4.astype(np.float32)
    print('Information of the dataframe for the modelling:\n')
    print(df4.info())
    
    # prepare the data for DDM
    xcols =  list(df4.columns.values)
    del xcols[0]   
    indexData = df4.index.values
    X = df4[xcols] #.values
    y = df4[f] #.values
    

    ## Training data size
    splits = TimeSeriesSplit(4) ## 3/4 for training and 1/4 for testing
    
    for trainIdx, testIdx in splits.split(X):
        trainIndex = trainIdx
        testIndex = testIdx
       

    X_train = X[:len(trainIndex)]
    X_test = X[len(trainIndex): (len(trainIndex)+len(testIndex))]
   
    y_train = y[:len(trainIndex)]
    y_test = y[len(trainIndex): (len(trainIndex)+len(testIndex))]
   
    print('Observations: %d' % (len(X_train)+len(X_test)))
    print('Training Observations: %d' % len(X_train))
    print('Testing Observations: %d' % len(X_test))

      ## normilize the input features      
    minMaxScaler = MinMaxScaler(feature_range=(0, 1))
    # X_train[X_train.columns] = minMaxScaler.fit_transform(X_train)
    # X_test[X_test.columns] = minMaxScaler.transform(X_test)
    X_train = minMaxScaler.fit_transform(X_train)
    X_test = minMaxScaler.transform(X_test)
    
    file = open(name+ "RF_parameters.txt", "w")
    
    def evaluate(model, test_features, test_labels):
                predictions = model.predict(test_features)
                errors = abs(predictions - test_labels)
                mape = 100 * np.mean(errors / test_labels)
                accuracy = 100 - mape
                print('Model Performance')
                print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
                print('Accuracy = {:0.2f}%.'.format(accuracy))
                
                return accuracy
    
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(random_state = 42)
    
    from sklearn.model_selection import RandomizedSearchCV

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    pprint(random_grid)
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor(random_state = 42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                  n_iter = 100, scoring='neg_mean_absolute_error', 
                                  cv = 3, verbose=2, random_state=42, n_jobs=-1,
                                  return_train_score=True)
    
    # Fit the random search model
    rf_random.fit(X_train, y_train);
    
    #cell 15
    rf_random.best_params_
    print('Best Parameters: ',rf_random.best_params_)
    
    file.write("Random search Best Parameters:")
    file.write(str(rf_random.best_params_))

    #cell 16
    rf_random.cv_results_

    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)
    
   
    #### Evaluate the Best Random Search Model
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)
    
    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
    
    # # Grid Search 

    # from sklearn.model_selection import GridSearchCV
    
    # # Create the parameter grid based on the results of random search 
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [80, 90, 100, 110],
    #     'max_features': [2, 3],
    #     'min_samples_leaf': [3, 4, 5],
    #     'min_samples_split': [8, 10, 12],
    #     'n_estimators': [100, 200, 300, 1000]
    # }
    
    # # Create a base model
    # rf = RandomForestRegressor(random_state = 42)
    
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
    #                           cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

    # # Fit the grid search to the data
    # grid_search.fit(X_train, y_train);
    # grid_search.best_params_
    
    # #### Evaluate the Best Model from Grid Search
    # best_grid = grid_search.best_estimator_
    # grid_accuracy = evaluate(best_grid, X_test, y_test)
    # print('Best Parameters Gread Search: ',grid_search.best_params_)
    
    # print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    # file.write("Gread search Best Parameters:")
    # file.write(str(grid_search.best_params_))
    file.close()
    gc.collect()  # Manual garbage collection 

#%%
