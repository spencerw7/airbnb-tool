import streamlit as st
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
import math

#import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, MinMaxScaler, StandardScaler
from analytics import load_data, est_occ_rate

AMENITIES = ["air conditioning", "heating", "washer", "dryer", "kitchen", "parking", "wifi", "tv", "pool", "hot tub"]

def amenities_simplify(am_df):
    '''Clean amenities list by reducing list to set amenities'''
    am_list = []
    for x in am_df:
        amenities = x.strip("[]").replace('"', '').split(",")
        #am_list_row = [i for i in amenities if i.lower() in AMENITIES]
        am_list_row = [i for i in amenities]
        am_list.append(am_list_row)
    return am_list

def remove_outliers(df,columns,n_std):
    '''Removes rows where values specified in the columns are n standard deviations above the mean'''
    for col in columns:
        
        mean = df[col].mean()
        sd = df[col].std()
        
        df = df[(df[col] <= mean+(n_std*sd))]
        
    return df

def pred_interval(X_df, y_df, input_df, y_hat, alpha, linear_model):
    '''Calculates prediction interval for predicted value'''
    k = len(linear_model.named_steps['linearregression'].coef_)
    X_transf = linear_model.named_steps['columntransformer'].transform(X_df)
    X_transf_t = np.transpose(X_transf)
    input_transf = linear_model.named_steps['columntransformer'].transform(input_df)
    input_transf_t = np.transpose(input_transf)

    t_crit = stats.t.ppf(1-alpha/2, len(y_df) - (k +1))
    mse = mean_squared_error(y_df, linear_model.predict(X_df))

    statistic = np.sqrt(mse * (1 + np.matmul(np.matmul(input_transf, np.linalg.inv(np.matmul(X_transf_t, X_transf))), input_transf_t)))

    lower = y_hat - statistic
    upper = y_hat + statistic

    return (lower[0][0], upper[0][0])


st.title("Airbnb Analytics Tool")
st.write("""
### Enter property info to get a rental revenue estimation.
""")

# Set columns
col1, col2, col3 = st.columns((1,1,1))

# User input
with col1:
    city_name = st.selectbox("Choose City", ("Montreal", "New Brunswick", "Ottawa", "Quebec City", "Toronto", "Vancouver", "Victoria"))
    
    df_cached = load_data(city_name)
    
    nbhd = st.selectbox("Choose Neighbourhood", df_cached["neighbourhood_cleansed"].unique())
    property_type = st.selectbox("Property Type", ("Entire home/apt", "Private room", "Shared room"))
    guest_num = st.number_input("Number of guests", 0, 20)
    room_num = st.number_input("Number of bedrooms", 0, 10)
    bed_num = st.number_input("Number of beds", 0, 15)
    amenities = st.multiselect("Enter amenities", [x.title() for x in AMENITIES])

    df = df_cached[["neighbourhood_cleansed", "room_type", "accommodates", "bedrooms", "beds", "amenities", "price"]]
    #df["amenities"] = amenities_simplify(df["amenities"])

    # Clean data and remove outliers
    df['price'] = df['price'].replace({'\$': '', ',': ''}, regex=True)
    df['price'] = df['price'].astype(float)
    df = remove_outliers(df,['price'],4)
    df = df[df['beds'] < 50]
    df = df[df['price'] != 0]
    df = df[~df['bedrooms'].isna()]

    st.write('\n\n\n')
    st.info('This is for educational purposes only and does not take into consideration any fees or additional costs.')
    
    ### ML ###
    X = df.drop(['price', 'amenities'], axis=1)
    y = np.log(df['price'])

    # get the categorical and numeric column names
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    # cat_cols.remove("amenities")
    # multi_cat_col = ["amenities"]

    # pipeline for numerical columns
    num_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        MinMaxScaler()
    )
    # pipeline for categorical columns
    cat_pipe = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='N/A'),
        OneHotEncoder(handle_unknown='ignore', sparse=False)
    )

    # pipeline for multi label column
    # multi_cat_pipe = make_pipeline(
    #     SimpleImputer(strategy='constant', fill_value='N/A'),
    #     MultiLabelBinarizer()
    # )

    # combine all the pipelines
    full_pipe = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    reg = make_pipeline(full_pipe, LinearRegression())
    reg.fit(X, y)
    
    if st.button('Enter'):
        with col2:
            # Store user input as a one row dataframe
            input_df = pd.DataFrame([[nbhd, property_type, guest_num, room_num, bed_num]], columns=X.columns)
            result = round(np.exp(reg.predict(input_df))[0], 2)

            st.title("Estimated monthly revenue:")
            prediction_interval = pred_interval(X, y, input_df, result, 0.05, reg)
            lower, upper = tuple([30*float(est_occ_rate.strip('%'))/100*x for x in prediction_interval])
            lower = '${:,.0f}'.format(int(lower/100)*100)
            upper = '${:,.0f}'.format(int(math.ceil(upper/100.0))*100)

            output = lower + ' to ' + upper + ' per month'

            st.metric('', output)


    