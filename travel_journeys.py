from utils import print_city_map
from utils import extract_coords
import matplotlib.pyplot as plt
import pandas as pd
import pickle


def plot_travel_journeys(df, map):

   #  Store the coordinates of the previous location
    prev_lat = None
    prev_long = None
    for index, row in df.iterrows():

        lat = row['latitude']
        long = row['longitude']

        if prev_lat is not None and prev_long is not None:
            # Plot a line between current and previous locations
            line = plt.Line2D([prev_lat, lat], [prev_long, long], color='red', alpha=0.002)
            map.add_artist(line)

        # Update previous location
        prev_lat = lat
        prev_long = long

      # Add circle to map on the last iteration
    circle = plt.Circle((prev_lat, prev_long), radius=20, color='black', alpha=1.0)
    map.add_patch(circle)


fig, (ax3) = plt.subplots( figsize=(10, 8))
ax3 = print_city_map(ax3)
ax3.set_title('Travel journeys to social gatherings for 1 week period')

with open("result_dict.pickle", "rb") as f:
    result_dict = pickle.load(f)

# Access the data in the result_dict
for hash_id, stamps in result_dict.items():
    # Perform operations on the stamps data
    stamps['longitude'], stamps['latitude'] = zip(*stamps['currentLocation'].apply(extract_coords))
    plot_travel_journeys(stamps, ax3)

plt.show()


