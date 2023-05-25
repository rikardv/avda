import numpy as np
from utils import print_city_map
from utils import extract_coords
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from math import radians, cos, sin, asin, sqrt
from filter_travel_journal import generate_pickled_travel_journal


'''
    Select participants 
'''
participant_a = 34
participant_b = 922

def plot_at_work_locations(df, color, ax):
    global plotted_coordinates
    categories = {
        'AtWork': 'Work',
        'AtHome': 'Home',
        'AtRestaurant': 'Restaurant',
        'AtRecreation': 'Socialize'
    }

    threshold_distance = 200

    def is_close_to_plotted_coordinates(latitude, longitude):
        for plotted_latitude, plotted_longitude in plotted_coordinates:
            distance = sqrt((latitude - plotted_latitude)**2 + (longitude - plotted_longitude)**2)
            if distance < threshold_distance:
                return True
        return False

    for category, label in categories.items():
        category_df = df[df['currentMode'] == category]
        if category_df.shape[0] > 0:
            unique_latitudes = category_df['latitude'].unique()
            unique_longitudes = category_df['longitude'].unique()
            for latitude, longitude in zip(unique_latitudes, unique_longitudes):
                if (latitude, longitude) not in plotted_coordinates:
                    if not is_close_to_plotted_coordinates(latitude, longitude):
                        ax.text(latitude, longitude, label, fontsize=10, color='black', alpha=0.9)
                    circle = plt.Circle((latitude, longitude), 50, color=color, fill=False)
                    ax.add_patch(circle)
                    plotted_coordinates.add((latitude, longitude))

def plot_travel_journeys(df, map, color):
    # Store the coordinates of the previous location
    prev_lat = None
    prev_long = None
    
    for index, row in df.iterrows():
        lat = row['latitude']
        long = row['longitude']

        if prev_lat is not None and prev_long is not None:

            # Plot a line between current and previous locations
            line = plt.Line2D([prev_lat, lat], [prev_long, long], color=color, alpha=0.25)
            map.add_artist(line)

        # Update previous location
        prev_lat = lat
        prev_long = long


max_retries = 2
retry_count = 0

while retry_count < max_retries:
    try:
        with open(f'{participant_a}_journeys.pickle', 'rb') as f:
            result_dict_a = pickle.load(f)
        break  # File opened successfully, exit the loop
    except FileNotFoundError:
        if retry_count < max_retries - 1:
            print(f'Pickled file not found - generating new one for participant {participant_a}')
            generate_pickled_travel_journal(participant_a)
            retry_count += 1
        else:
            print(f'Failed to open pickled file after {max_retries} attempts. Aborting.')
            exit()
            # Handle the failure, raise an exception or exit the program

retry_count = 0  # Reset the retry count for participant B

while retry_count < max_retries:
    try:
        with open(f'{participant_b}_journeys.pickle', 'rb') as f:
            result_dict_b = pickle.load(f)
        break  # File opened successfully, exit the loop
    except FileNotFoundError:
        if retry_count < max_retries - 1:
            print(f'Pickled file not found - generating new one for participant {participant_b}')
            generate_pickled_travel_journal(participant_b)
            retry_count += 1
        else:
            print(f'Failed to open pickled file after {max_retries} attempts. Aborting.')
            exit()
            # Handle the failure, raise an exception or exit the program

# Get the participants apartment ids
df_participants_data = pd.read_csv('./csv/participant_data.csv')

# Get the apartments location
df_apartments_locations = pd.read_csv('./csv/Apartments.csv')

df_buildins_locations = pd.read_csv('./csv/Buildings.csv')

# Get the apartments location
df_apartments_locations['longitude'], df_apartments_locations['latitude'] = zip(*df_apartments_locations['location'].apply(extract_coords))
df_buildins_locations['longitude'], df_buildins_locations['latitude'] = zip(*df_buildins_locations['location'].apply(extract_coords))

apartment_id_a = df_participants_data.loc[df_participants_data['participantId'] == participant_a, 'apartmentId']
apartment_id_b = df_participants_data.loc[df_participants_data['participantId'] == participant_b, 'apartmentId']


apartment_location_a_lat = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_a.iloc[0], 'latitude']
apartment_location_a_long = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_a.iloc[0], 'longitude']

apartment_location_b_lat = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_b.iloc[0], 'latitude']
apartment_location_b_long = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_b.iloc[0], 'longitude']

title = f'Participant {participant_a} (red) and {participant_b} (blue) travel journeys'

fig, (ax3) = plt.subplots( figsize=(15, 8))
ax3 = print_city_map(ax3)
ax3.set_title(title)

plotted_coordinates = set()

# Access the data in the result_dict
for hash_id, stamps in result_dict_a.items():
    # Perform operations on the stamps data
    stamps['longitude'], stamps['latitude'] = zip(*stamps['currentLocation'].apply(extract_coords))
    color = 'red'
    plot_travel_journeys(stamps, ax3, color)
    plot_at_work_locations(stamps, color, ax3)


# Access the data in the result_dict
for hash_id, stamps in result_dict_b.items():
    # Perform operations on the stamps data
    stamps['longitude'], stamps['latitude'] = zip(*stamps['currentLocation'].apply(extract_coords))
    color = 'blue'
    plot_travel_journeys(stamps, ax3, color)
    plot_at_work_locations(stamps, color, ax3)



plt.show()



