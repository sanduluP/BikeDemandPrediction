# BikeDemandPrediction

demandprediction.py is the main file to be executed. 

Config file consists of user input details. 

XGB_Regressor.py file consists of model-related content.

    Ridge :                 ['MAE':114.435, 'R2': 0.414]
    Lasso :                 ['MAE':124.647, 'R2': 0.31]
    SVR :                   ['MAE':161.582, 'R2': -0.128]
    KNeighborsRegressor :   ['MAE':119.407, 'R2': 0.364]
    DecisionTreeRegressor : ['MAE':100.491, 'R2': 0.57]
    RandomForestRegressor : ['MAE':98.82,   'R2': 0.6]
    XGBRegressor :          ['MAE':95.892, 'R2': 0.629]

    XGBRegressor is promising model. With hyper parameter tuning, final result is

    XGBRegressor With Best Parameters : ['MAE':93.719, 'R2': 0.645]
