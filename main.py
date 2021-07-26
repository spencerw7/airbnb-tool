from sys import excepthook
from numpy.core.fromnumeric import clip
from pandas.io.parsers import read_csv
from seaborn.rcmod import axes_style
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

# Format prices
try:
    prices = pd.to_numeric(df['price'].replace({'\$':'',',':''}, regex=True).astype(float))
except KeyError:
    prices = 0
df['price'] = prices
# Filter out outliers
df = df[df['price'] < 3000]    

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

avg_price = '${:,.2f}'.format(df['price'].mean())
est_occ_rate = '{:.1%}'.format(df['availability_90'].mean()/90)

# Get number of amenities
ac = 0
heating = 0
washer = 0
dryer = 0
kitchen = 0
parking = 0
internet = 0
tv = 0
pool = 0
hottub = 0
for x in df['amenities']:
    amenities = x.strip("[]").replace('"', '').split(",")
    for phrase in amenities:
        if "Air conditioning" in phrase:
            ac += 1
        if "Heating" in phrase:
            heating += 1
        if "Washer" in phrase:
            washer += 1
        if "Dryer" in phrase:
            dryer += 1
        if "Kitchen" in phrase:
            kitchen += 1
        if "Free parking on premises" in phrase:
            parking += 1
        if "Wifi" in phrase:
            internet += 1
        if "TV" in phrase:
            tv += 1
        if "Pool" in phrase:
            pool += 1
        if "Hot tub" in phrase:
            hottub += 1

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
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    sns.set_palette("pastel")
    ax = sns.barplot(x=df['accommodates'],y=df['price'], estimator=np.mean, ci=None, edgecolor=None)
    ax.set(xlabel='Maximum Guests', ylabel='Nightly Price ($)', title='Avg. Nightly Price by Number of Guests')
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

    fig, ax = plt.subplots(figsize=(9.1,1))
    ax.set_visible(False)
    fig.patch.set_alpha(0.1)
    st.pyplot(fig)

    # Amenities plot
    fig, ax = plt.subplots(1,1)
    amenities_df = pd.DataFrame({"type":["Air conditioning", "Heating", "Washer", "Dryer", "Kitchen", "Parking", "Internet", "TV", "Pool", "Hot tub"],
                                "pct":[ac/len(df["amenities"]), heating/len(df["amenities"]), washer/len(df["amenities"]), dryer/len(df["amenities"]), kitchen/len(df["amenities"]), parking/len(df["amenities"]), internet/len(df["amenities"]), tv/len(df["amenities"]), pool/len(df["amenities"]), hottub/len(df["amenities"])]})
    ax = sns.barplot(x="pct", y="type", data=amenities_df)
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set(xlabel=None, ylabel=None)
    plt.title("Amenities")
    st.pyplot(fig)

    # Price distribution plot
    fig, ax = plt.subplots(1,1)
    ax = sns.kdeplot(x=df["price"], hue=df["host_is_superhost"], common_norm=True, fill=True, legend=False, clip=(-100,600))
    plt.title("Price Density of Superhosts vs. Non Superhosts")
    plt.legend(title='Superhost', loc='upper right', labels=['Yes', 'No'])
    plt.yticks([])
    ax.set(xlabel="Price ($)", ylabel=None)
    st.pyplot(fig)