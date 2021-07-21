from sys import excepthook
from pandas.io.parsers import read_csv
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# Main title
st.title("Airbnb Analytics Tool")
st.write("""
### Analysis of Airbnb listings in Canadian cities.
""")

# Set columns
col1, col2, col3 = st.beta_columns((2,1,1))

with col1:
    city_name = st.selectbox("Choose City", ("Montreal", "New Brunswick", "Ottawa", "Quebec City", "Toronto", "Vancouver", "Victoria"))


def header(url):
     st.markdown(f'<p style="background-color:#434c5e;color:#fffff;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

# Set colour on map based on type of rental
def colour_picker(data):
    if data == "Entire home/apt":
        colour = [200, 30, 0, 160]
    elif data == "Private room":
        colour = [0, 200, 30, 160]
    elif data == "Shared room":
        colour = [255, 255, 0, 160]
    elif data == "Hotel room":
        colour = [0, 50, 50, 160]
    
    return colour

# Cache data for faster loading times when switching back and forth between cities
@st.cache
def load_data(city):
    if city == "Montreal":
        url = "http://data.insideairbnb.com/canada/qc/montreal/2021-04-17/data/listings.csv.gz"
    elif city == "New Brunswick":
        url = "http://data.insideairbnb.com/canada/nb/new-brunswick/2021-01-30/data/listings.csv.gz"
    elif city == "Ottawa":
        url = "http://data.insideairbnb.com/canada/on/ottawa/2021-04-19/data/listings.csv.gz"
    elif city == "Quebec City":
        url = "http://data.insideairbnb.com/canada/qc/quebec-city/2021-04-11/data/listings.csv.gz"
    elif city == "Toronto":
        url = "http://data.insideairbnb.com/canada/on/toronto/2021-04-09/data/listings.csv.gz"
    elif city == "Vancouver":
        url = "http://data.insideairbnb.com/canada/bc/vancouver/2021-04-12/data/listings.csv.gz"
    elif city == "Victoria":
        url = "http://data.insideairbnb.com/canada/bc/victoria/2021-03-27/data/listings.csv.gz"

    data = read_csv(url)
    return data

data_load_state = st.text('Loading data...')
df_cached = load_data(city_name)
data_load_state.text("")

# Get neighbourhoods
nbhds = df_cached["neighbourhood_cleansed"].unique()
nbhds = np.insert(nbhds, 0, "All")

with col2:
    neighbourhood = st.selectbox("Filter by Neighbourhood", nbhds)
 
# Create copy of dataframe to allow mutations while still being able to cache
df_unfiltered = df_cached.copy()

# Filter data on neighbourhood
if neighbourhood == "All":
    df = df_unfiltered
else:
    df = df_unfiltered[df_unfiltered["neighbourhood_cleansed"] == neighbourhood]

df['colour'] = df['room_type'].apply(colour_picker)
midpoint = (np.average(df["latitude"]), np.average(df["longitude"]))

# Calculate metrics
try:
    entire_home_pct = '{:.1%}'.format(df['room_type'].value_counts()['Entire home/apt']/len(df['room_type']))
except KeyError:
    entire_home_pct = '{:.1%}'.format(0)
try:
    private_room_pct = '{:.1%}'.format(df['room_type'].value_counts()['Private room']/len(df['room_type']))
except KeyError:
    private_room_pct = '{:.1%}'.format(0)
try:
    shared_room_pct = '{:.1%}'.format(df['room_type'].value_counts()['Shared room']/len(df['room_type']))
except KeyError:
    shared_room_pct = '{:.1%}'.format(0)
try:
    prices = pd.to_numeric(df['price'].replace({'\$':'',',':''}, regex=True).astype(float))
except KeyError:
    prices = 0
avg_price = '${:,.2f}'.format(prices.mean())
est_occ_rate = '{:.1%}'.format(df['availability_90'].mean()/90)

with col1:
    # Interactive map
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=10,
            pitch=0
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df[['latitude', 'longitude', 'room_type', 'colour']],
                get_position='[longitude, latitude]',
                get_fill_color='colour',
                get_radius=50
            )
        ]
    ))
    # Bar plot for average nightly prices by number of guests
    fig, ax = plt.subplots()
    ax = sns.barplot(x=df['accommodates'],y=prices, estimator=np.mean, ci=None)
    ax.set(xlabel='Maximum Guests', ylabel='Nightly Price ($)')
    plt.title("Avg. Nightly Price by Number of Guests")
    ax.set_facecolor("#273346")
    fig.patch.set_facecolor("#273346")
    st.pyplot(fig)

with col2:
    # Figures to show percentages of rental types, average nightly price, and occupancy rate
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.text(0, 0.9, "Entire home/apt: " + str(entire_home_pct) + "\nPrivate Room: " + str(private_room_pct) + "\nShared Room: " + str(shared_room_pct), 
        fontsize=24, color='w',
         ha="left", va="top",
         bbox=dict(boxstyle="round",
                   ec='None',
                   fc='#273346',
                   )
         )

    ax.text(0, 0.5, avg_price + "\nAvg. Nightly Price", fontsize=40, color='w',
         ha="left", va="top",
         bbox=dict(boxstyle="round",
                   ec='None',
                   fc='#273346',
                   )
         )

    ax.text(0, 0.075, est_occ_rate + "\nEst. Occupancy", fontsize=40, color='w',
         ha="left", va="top",
         bbox=dict(boxstyle="round",
                   ec='None',
                   fc='#273346',
                   )
         )
    
    circle_red = plt.Circle((0.9,0.87), 0.045, fc = "#bf616a", ec="None")
    circle_green = plt.Circle((0.9,0.77), 0.045, fc = "#a3be8c", ec="None")
    circle_yellow = plt.Circle((0.9,0.67), 0.045, fc = "#ebcb8b", ec="None")
    ax.add_artist(circle_red)
    ax.add_artist(circle_green)
    ax.add_artist(circle_yellow)

    ax.axis('off')
    st.pyplot(fig)
