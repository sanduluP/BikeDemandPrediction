from load_data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from XGB_Regressor import XBGRegressor


def check_null_value(data):
    return str(data.isnull().sum().sum()), str(data.isna().sum().sum())


def drop_column_data(data, col_name):
    data.drop(col_name, axis=1, inplace=True)


def print_header(title):
    print("\n")
    print("---", title, "---")
    print("\n")


# column value Analysis
def group_by(col):
    print("\n")
    print("--- ", col, " analysis", " ---")
    df_summary = pd.DataFrame()
    df_summary["Count"] = (train_df.groupby([col])[col].count())
    df_summary["AverageCount"] = (train_df.groupby([col])['cnt'].mean())
    print(df_summary)
    frame = pd.DataFrame({"count": train_df["cnt"], col: train_df[col]})
    return frame.groupby(col).groups


def get_vif(data):
    vif = pd.DataFrame()
    vif["variables"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif


def get_metrics(y_true, y_pred, model_name):
    # MSE = mean_squared_error(y_true, y_pred)
    # RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    print(f"{model_name} : ['MAE':{round(MAE, 3)}, 'R2': {round(R2, 3)}]")


if __name__ == '__main__':
    dataloader = DataLoader("hour", randomState=23)
    df, train_df, test_df = dataloader.fetchData()
    print("Shape of dataset: ", df.shape)

    # drop ID Column
    drop_column_data(df, "instant")

    # finding NAs and NULLs
    print("Null and NA value for train set: ", check_null_value(df))

    ## Data Preprocessing
    # Info about dataframes
    print("Information about data:", df.info())

    # dteday column as a whole is not useful. Extract day from it and drop the dteday column
    df['dteday'] = pd.to_datetime(df['dteday'])
    df["Day"] = df["dteday"].dt.day

    train_df['dteday'] = pd.to_datetime(train_df['dteday'])
    train_df["Day"] = train_df["dteday"].dt.day

    test_df['dteday'] = pd.to_datetime(test_df['dteday'])
    test_df["Day"] = test_df["dteday"].dt.day
    # drop date column
    drop_column_data(df, "dteday")
    drop_column_data(train_df, "dteday")
    drop_column_data(test_df, "dteday")

    category_features = ['season', 'yr', 'mnth', 'Day', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
    target_feature = ['cnt']
    features = category_features + numeric_features

    # convert type to category
    for col in category_features:
        df[col] = df[col].astype('category')
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')
    print(df[category_features].describe())

    ## NUMERICAL DATA EXPLORATION
    print_header("Analysis 1 || Projection || has been started")

    groups = group_by('temp')
    print("Since we saw that 'temp' is important for our analysis, as the mean of groups vary from each other."
          "we decided to keep temp column during analysis")

    groups = group_by('atemp')
    print("Since we saw that 'atemp' is important for our analysis, as the mean of groups vary from each other."
          "we decided to keep atemp column during analysis")

    groups = group_by('hum')
    print("Since we saw that 'hum' is important for our analysis, as the mean of groups vary from each other."
          "we decided to keep hum column during analysis")

    groups = group_by('windspeed')
    print("Since we saw that 'windspeed' is important for our analysis, as the mean of groups vary from each other."
          "we decided to keep windspeed column during analysis")

    print_header("Analysis 1 || Projection || has been ended")
    print_header("Analysis 2 || Data Preprocessing - Manipulation || has been started")

    print(df[numeric_features].describe().T)
    # values are scaled. Hence range is from 0-1

    sns.pairplot(df[numeric_features + target_feature])

    working_day_data = train_df[train_df['workingday'] == 0]
    holiday_data = train_df[train_df['workingday'] == 1]

    ## Categorical Features
    # Visualization for analysing outliers
    sns.set(font_scale=1.0)
    fig, axes = plt.subplots(nrows=4, ncols=2)
    fig.set_size_inches(25, 25)
    sns.boxplot(data=train_df, y="cnt", x="workingday", orient="v", ax=axes[0][0])
    sns.boxplot(data=train_df, y="cnt", x="mnth", orient="v", ax=axes[0][1])
    sns.boxplot(data=working_day_data, y="cnt", x="hr", orient="v", ax=axes[1][0])
    sns.boxplot(data=holiday_data, y="cnt", x="hr", orient="v", ax=axes[1][1])
    sns.boxplot(data=train_df, y="cnt", x="season", orient="v", ax=axes[2][0])
    sns.boxplot(data=train_df, y="cnt", x="weathersit", orient="v", ax=axes[2][1])
    sns.boxplot(data=train_df, y="cnt", x="temp", orient="v", ax=axes[3][0])
    sns.boxplot(data=train_df, y="cnt", orient="v", ax=axes[3][1])

    axes[0][0].set(xlabel='Working Day', ylabel='Count', title="Box plot on Count across working days")
    axes[0][1].set(xlabel='Month', ylabel='Count', title="Box plot on Count across Months")
    axes[1][0].set(xlabel='Hour Of The Holiday', ylabel='Count', title="Box plot on Count across hour of week holiday")
    axes[1][1].set(xlabel='Hour Of The Weekday', ylabel='Count', title="Box plot on Count across hour of week day")
    axes[2][0].set(xlabel='Seasons', ylabel='Count', title="Box plot on Count across season")
    axes[2][1].set(xlabel='Weather Situation', ylabel='Count', title="Box plot on Count across weather situations")
    axes[3][0].set(xlabel='Temperature', ylabel='Count', title="Box plot on Count across temperature")
    axes[3][1].set(ylabel='Count', title="Box Plot On Count")

    '''
    __Analysis Summary:__ 
    Upon thorough examination of the data, distinct patterns emerge between working days and holidays. 
    It is evident that on regular working days, there is a substantial surge in bicycle rentals compared to holidays. 
    Notably, peak demand occurs at 8 AM and 5 PM, indicative of commuters relying on the service for work or school 
    transportation. Conversely, during holidays, the peak demand period shifts between 12 PM and 5 PM.
    
    Temperature plays a pivotal role in influencing bike rental trends.Lower temperatures not only reduce the average rentals 
    but also reveal more outlier data points. 
    Moreover, it is worth noting that during the summer season and under clear or partly cloudy weather conditions, demand experiences a notable increase.
    This analysis underscores the importance of weather conditions and day type in predicting bicycle rental patterns, providing valuable insights for optimizing service offerings.
    '''

    ## Target feature
    sns.displot(train_df[target_feature])

    # plot is right skewed. We will use inter quartile range to identify and remove outliers

    print("Count in train set with outliers: {}".format(len(train_df)))
    q1 = train_df.cnt.quantile(0.25)
    q3 = train_df.cnt.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    train_df = train_df.loc[(train_df.cnt >= lower_bound) & (train_df.cnt <= upper_bound)]
    print("Count in train set without outliers: {}".format(len(train_df)))
    sns.displot(np.sqrt(train_df.cnt))

    # variable with value 0 is not skewed, its normally distributed, minus is left skewed, positive is right skewed
    # hum,atemp,temp are left skewed...tail towards left.
    # windspeed is right skewed: from pairplot histogram we can visualize
    print(train_df[numeric_features + target_feature].skew().sort_values(ascending=True))
    print_header("Analysis 2 || Data Preprocessing - Manipulation || has been ended")
    print_header("Analysis 3 || Correlation || has been started")
    # Correlation Tests
    plt.figure(figsize=(10, 10))
    sns.heatmap(train_df[numeric_features + target_feature].corr(), annot=True, cmap="coolwarm",
                square=True)  # pearson co relation
    '''
    Casual & registered have direct information about bike count, which is to predict. Presence of these feature is data leakage
    Hence we need to remove.
    Dependent variable cnt with respect to independent features are not highly colinear.
    In perspective of multicolinearity dependent features temp and atemp are highly related. Deciding which one to remove,
    let check the matrix value wrt to cnt, again both of value is 0.43.We need to check further, in terms of VIF(variance influence factor)
    '''
    print("Variance Influence Factor Analysis")
    print(get_vif(train_df[[i for i in train_df.describe().columns if i in numeric_features]]))
    # atemp has more VIF, hence we can remove temp

    drop_column_data(train_df, ["instant", "casual", "registered", "temp"])
    drop_column_data(test_df, ["instant", "casual", "registered", "temp"])
    plt.show()
    print_header("Analysis 3 || Correlation || has been ended")
    print_header("Encoding has been started")

    train_df = pd.DataFrame(pd.get_dummies(data=train_df, columns=['season',
                                                                   'weekday',
                                                                   'weathersit',
                                                                   'holiday',
                                                                   'mnth',
                                                                   'hr',
                                                                   'workingday',
                                                                   'Day', 'yr'], drop_first=True, dtype=bool))

    test_df = pd.DataFrame(pd.get_dummies(data=test_df, columns=['season',
                                                                 'weekday',
                                                                 'weathersit',
                                                                 'holiday',
                                                                 'mnth',
                                                                 'hr',
                                                                 'workingday',
                                                                 'Day', 'yr'], drop_first=True, dtype=bool))

    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)
    missing_cols = train_columns - test_columns
    # Add the missing columns to test_df with all values set to 0
    for col in missing_cols:
        test_df[col] = 0
    test_df = test_df[train_df.columns]

    print_header("Encoding has been ended")
    print_header("Analysis 4 || Modelling is started")
    features = train_df.columns.drop(target_feature)
    X_train = train_df[features].values
    y_train = train_df[target_feature].values.ravel()
    features = test_df.columns.drop(target_feature)
    X_test = test_df[features].values
    y_test = test_df[target_feature].values.ravel()

    models = [
        Ridge(),
        Lasso(),
        SVR(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        XGBRegressor()
    ]

    for model in models:
        model_fit = model.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)
        get_metrics(y_test, y_pred, type(model).__name__)

    '''
    Ridge :                 ['MAE':114.435, 'R2': 0.414]
    Lasso :                 ['MAE':124.647, 'R2': 0.31]
    SVR :                   ['MAE':161.582, 'R2': -0.128]
    KNeighborsRegressor :   ['MAE':119.407, 'R2': 0.364]
    DecisionTreeRegressor : ['MAE':100.491, 'R2': 0.57]
    RandomForestRegressor : ['MAE':98.82, 'R2': 0.6]
    XGBRegressor :          ['MAE':95.892, 'R2': 0.629]
    
    XGBRegressor has better R2 value, we are proceeding with this for fine-tuning
    '''
    xgb = XBGRegressor()
    cv = xgb.randomParameterSearch()
    cv.fit(X_train,y_train)
    y_pred_xgb_random = cv.predict(X_test)
    get_metrics(y_test, y_pred_xgb_random, "XGBRegressor With Best Possible Parameters")
    print("Best parameters: ", cv.best_params_)
    '''
    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    XGBRegressor With Best Parameters : ['MAE':93.719, 'R2': 0.645]
    Best parameters:  {'subsample': 0.7999999999999999, 'n_estimators': 1000, 'max_depth': 5, 
    'learning_rate': 0.2, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7}
    '''
    xgbr = xgb.trainBestParam(cv.best_params_)
    xgbr.fit(X_train, y_train)
    y_pred_tuned = xgbr.predict(X_test)
    get_metrics(y_test, y_pred_tuned, "XGBRegressor With Best Parameters")

    # XGBRegressor With Best Parameters : ['MAE':93.719, 'R2': 0.645]
    print_header("Modelling has been ended")