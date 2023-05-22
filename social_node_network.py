import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from shapely.geometry import Point
import pandas as pd
from utils import *

# The interactions between participants
df = pd.read_csv('./csv/socials.csv')

# Convert the 'timestamp' column to datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the data based on the date range
start_date = pd.to_datetime('2022-03-01T00:00:00Z')
end_date = pd.to_datetime('2022-03-06T00:00:00Z')
df_filtered_interactions = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

# Get the participants apartment ids
df_participants_apartments = pd.read_csv('./csv/participant_data.csv')

# Get the apartments location
df_apartments_locations = pd.read_csv('./csv/Apartments.csv')

# Cluster the apartments locations in 4 regions
from sklearn.cluster import KMeans

# Apply the function to the location column to extract the latitude and longitude values
df_apartments_locations['longitude'], df_apartments_locations['latitude'] = zip(*df_apartments_locations['location'].apply(extract_coords))
kmeans = KMeans(n_clusters=8, random_state=0).fit(df_apartments_locations[['latitude', 'longitude']])
df_apartments_locations['region'] = kmeans.labels_

## Elbow graph
# ssd = []
# for k in range(1, 16):
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(df_apartments_locations[['latitude', 'longitude']])
#     ssd.append(kmeans.inertia_)

# #Plot the elbow graph
# plt.plot(range(1, 16), ssd, 'bx-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of squared distances')
# plt.title('Elbow graph for check-in clustering')
# plt.show()


# Create a dictionary to map integer values to colors
color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'purple', 5: 'yellow', 6: 'cyan', 7: 'magenta'}



def plot_social_node(df,map):
    contacts_outside_region = 0
    contacts_inside_region = 0
    for _, row in df_filtered_interactions.iterrows():
        participant_id_from = row['participantIdFrom']
        apartment_id_from = df_participants_apartments.loc[df_participants_apartments['participantId'] == participant_id_from, 'apartmentId']
        if apartment_id_from.empty:
            continue
        apartment_location_from_lat = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_from.iloc[0], 'latitude']
        apartment_location_from_long = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_from.iloc[0], 'longitude']
        apartment_region_from = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_from.iloc[0], 'region']

        participant_id_to = row['participantIdTo']
        apartment_id_to = df_participants_apartments.loc[df_participants_apartments['participantId'] == participant_id_to, 'apartmentId']
        if apartment_id_to.empty:
            continue
        apartment_location_to_lat = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_to.iloc[0], 'latitude']
        apartment_location_to_long = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_to.iloc[0], 'longitude']
        apartment_region_to = df_apartments_locations.loc[df_apartments_locations['apartmentId'] == apartment_id_to.iloc[0], 'region']

        if apartment_region_from.iloc[0] == apartment_region_to.iloc[0]:
            contacts_inside_region += 1
            edge_color = color_map[apartment_region_from.iloc[0]]
        else:
            contacts_outside_region += 1
            edge_color = 'gray'
         # Create a line between the coordinates
        line = plt.Line2D([apartment_location_from_lat, apartment_location_to_lat], [apartment_location_from_long, apartment_location_to_long], color=edge_color, alpha=0.03)
        map.add_line(line)
        cirle = plt.Circle((apartment_location_from_lat, apartment_location_from_long), 20, color=color_map[apartment_region_from.iloc[0]], alpha=1.0)
        map.add_patch(cirle)
    return contacts_inside_region, contacts_outside_region


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax = print_city_map(ax)
contacts_inside_region, contacts_outside_region = plot_social_node(df_apartments_locations, ax)
print("Contacts inside region: ", contacts_inside_region)
print("Contacts outside region: ", contacts_outside_region)
print("Outside region percentage: ", contacts_outside_region/(contacts_inside_region+contacts_outside_region)*100, "%")
print("Inside region percentage: ", contacts_inside_region/(contacts_inside_region+contacts_outside_region)*100, "%")
plt.show()