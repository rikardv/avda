import numpy as np
from utils import print_city_map
from utils import extract_coords
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from math import radians, cos, sin, asin, sqrt
from filter_travel_journal import generate_pickled_travel_journal

# Get the 'RdYlGn' colormap
cmap = plt.cm.get_cmap('cool')


def calculate_distance(df):
    # Store the coordinates of the previous location
    prev_lat = None
    prev_long = None
    max_distance = 0
    min_distance = float('inf')
    
    for index, row in df.iterrows():
        lat = row['latitude']
        long = row['longitude']

        if prev_lat is not None and prev_long is not None:
            # Calculate the distance between current and previous locations
            # euklidian_distance = sqrt((lat - prev_lat) ** 2 + (long - prev_long) ** 2)
            manhattan_distance = abs(lat - prev_lat) + abs(long - prev_long)

            # Update the maximum distance if necessary
            if manhattan_distance > max_distance:
                max_distance = manhattan_distance

            # Update the minimum distance if necessary
            if manhattan_distance < min_distance:
                min_distance = manhattan_distance

        # Update previous location
        prev_lat = lat
        prev_long = long

    return max_distance, min_distance


def plot_travel_journeys(df, map, max_distance, min_distance, normalized_distances):
    # Store the coordinates of the previous location
    prev_lat = None
    prev_long = None
    
    for index, row in df.iterrows():
        lat = row['latitude']
        long = row['longitude']

        if prev_lat is not None and prev_long is not None:
            # Calculate the distance between current and previous locations
            # distance = sqrt((lat - prev_lat) ** 2 + (long - prev_long) ** 2)
            distance = abs(lat - prev_lat) + abs(long - prev_long)

            # Normalize the distance to a value between 0 and 1
            normalized_distance = (distance) / (max_distance)

            normalized_distances.append(normalized_distance)

            # Map the normalized distance to a value between 0 and 1
            color_value = normalized_distance

             # Get the color from the colormap based on the color value
            line_color = cmap(color_value)
            
            # Plot a line between current and previous locations
            line = plt.Line2D([prev_lat, lat], [prev_long, long], color=line_color, alpha=0.005)
            map.add_artist(line)

        # Update previous location
        prev_lat = lat
        prev_long = long

    # Add circle to map on the last iteration
    # circle = plt.Circle((prev_lat, prev_long), radius=20, color='black', alpha=1.0)
    # map.add_patch(circle)


purpose_dict = {
    "Purpose1": "Coming Back From Restaurant",
    "Purpose2": "Eating",
    "Purpose3": "Going Back to Home",
    "Purpose4": "Recreation (Social Gathering)",
    "Purpose5": "Work/Home Commute"
}

purpose = "Purpose3"

max_retries = 2
retry_count = 0
while retry_count < max_retries:
    try:
        with open(f"{purpose}_journeys.pickle", "rb") as f:
            result_dict = pickle.load(f)
            title = 'Travel journeys to eating locations for 1 week period'
        break
    except FileNotFoundError:
        if retry_count < max_retries - 1:
            print('File not found - generating new file')
            generate_pickled_travel_journal(False, purpose_dict[purpose], purpose)
            retry_count += 1
        else:
            print(f'Failed to open file {purpose}.pickle after {max_retries} retries')
            exit()

fig, (ax3, ax4) = plt.subplots( nrows=1, ncols=2,figsize=(20, 8))
ax3 = print_city_map(ax3)
ax3.set_title(title)

# Calculate the maximum and minimum distances
max_distance = 0
min_distance = float('inf')
normalized_distances = []

for hash_id, stamps in result_dict.items():
    # Perform operations on the stamps data
    stamps['longitude'], stamps['latitude'] = zip(*stamps['currentLocation'].apply(extract_coords))
    cur_max_distance, cur_min_distance = calculate_distance(stamps)
    max_distance = max(max_distance, cur_max_distance)
    min_distance = min(min_distance, cur_min_distance)

print("Max distance: ", max_distance)
print("Min distance: ", min_distance)
# Access the data in the result_dict
for hash_id, stamps in result_dict.items():
    # Perform operations on the stamps data
    stamps['longitude'], stamps['latitude'] = zip(*stamps['currentLocation'].apply(extract_coords))
    plot_travel_journeys(stamps, ax3, max_distance, min_distance, normalized_distances)

# Create a colorbar legend
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax3, label='Speed of travel')

ax4.hist(normalized_distances,bins=100, edgecolor='black', linewidth=1.2)
ax4.set_title('Distribution of travel speeds')

plt.show()



