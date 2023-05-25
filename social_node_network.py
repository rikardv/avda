import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from shapely.geometry import Point
import pandas as pd
from utils import *

# The interactions between participants
df = pd.read_csv('./csv/SocialNetwork.csv')

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
color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'purple', 5: 'cyan', 6: 'magenta', 7: 'pink'}

track_connections = {
    str(i): {str(j): 0 for j in range(8)} for i in range(8)
}

manhattan_distances = []

def plot_social_node(df,map):
    global manhattan_distances
    contacts_outside_region = np.zeros(8).astype(int)
    contacts_inside_region = np.zeros(8).astype(int)
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
            contacts_inside_region[apartment_region_from.iloc[0]] += 1
            edge_color = color_map[apartment_region_from.iloc[0]]
        else:
            contacts_outside_region[apartment_region_from.iloc[0]] += 1
            edge_color = 'gray'

        manhattan_distance = (abs(apartment_location_from_lat.values[0] - apartment_location_to_lat.values[0])) + (abs(apartment_location_from_long.values[0] - apartment_location_to_long.values[0]))
        manhattan_distances.append(manhattan_distance)

         # Create a line between the coordinates
        track_connections[str(apartment_region_from.iloc[0])][str(apartment_region_to.iloc[0])] += 1
        line = plt.Line2D([apartment_location_from_lat, apartment_location_to_lat], [apartment_location_from_long, apartment_location_to_long], color=edge_color, alpha=0.02)
        map.add_line(line)
        # cirle = plt.Circle((apartment_location_from_lat, apartment_location_from_long), 20, color=color_map[apartment_region_from.iloc[0]], alpha=0.0)
        # map.add_patch(cirle)
    return contacts_inside_region, contacts_outside_region


fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax = print_city_map(ax)
ax.set_title('Social interactions between participants')
# Get the cluster centroids
cluster_centroids = kmeans.cluster_centers_

# Print the cluster centroid coordinates
for i, centroid in enumerate(cluster_centroids):
    print(f"Cluster {i+1} centroid: Latitude {centroid[0]}, Longitude {centroid[1]}")
    ax.text(centroid[0], centroid[1], i, fontsize=18, color='black', alpha=0.9)


contacts_inside_region, contacts_outside_region = plot_social_node(df_apartments_locations, ax)

# Calculate the x-axis positions for the bars
x = np.arange(len(contacts_inside_region))

# Create the stacked bar chart
bar1 = ax1.bar(x, contacts_inside_region, label='Contacts inside region', color=[color_map[i] for i in range(len(contacts_inside_region))], alpha=0.5)
bar2 = ax1.bar(x, contacts_outside_region, bottom=contacts_inside_region, label='Contacts outside region', color='gray', alpha=0.5)

# Add labels and title
ax1.set_xlabel('Region')
ax1.set_ylabel('Number of contacts')
ax1.set_title('Number of contacts inside and outside the region')

# Add text inside the bars
for i, bar in enumerate(bar1):
    percentage = round(contacts_inside_region[i] / (contacts_inside_region[i] + contacts_outside_region[i]) * 100, 2)
    formatted_percentage = str(percentage) + '%'
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height / 2, formatted_percentage, ha='center', va='center', color='white')

for i, bar in enumerate(bar2):
    percentage = round(contacts_outside_region[i] / (contacts_inside_region[i] + contacts_outside_region[i]) * 100, 2)
    formatted_percentage = str(percentage) + '%'
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, formatted_percentage, ha='center', va='center', color='black')


# for region, occurrences in track_connections.items():
#     max_occurrences = max(occurrences.values())
#     max_region = max(occurrences, key=occurrences.get)
#     total_occurrences = sum(occurrences.values())
#     percentage = (max_occurrences / total_occurrences) * 100
#     print(f"Region {region} has the most visits to Region {max_region} with {percentage:.2f}%")
# ax1.legend()

# Plot histogram
# ax2.hist(manhattan_distances, bins=50, color='gray', alpha=0.9)
# # Set the number of x-axis ticks
# ax2.set_xticks(range(0, 12000, 1000))
# ax2.set_xlabel('Manhattan distance')
# ax2.set_ylabel('Number of contacts')


plt.show()