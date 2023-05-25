import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def plot_regions_on_map(df, map):
    region_df = df[['venueId', 'latitude', 'longitude', 'regions']]
    region_df.drop_duplicates(subset=['regions'], inplace=True)

    for index, row in region_df.iterrows():
        region = row['regions']
        latitudes = df[df['regions'] == region]['latitude']
        longitudes = df[df['regions'] == region]['longitude']
        mean_lat = latitudes.mean()
        mean_long = longitudes.mean()

        # Add circle to map with mean center and radius of 1000
        circle = plt.Circle((mean_lat, mean_long), radius=500, color=colors[int(region)], alpha=0.2)
        map.add_patch(circle)
        
        for index, point in df[df['regions'] == region].iterrows():
            point_lat = point['latitude']
            point_long = point['longitude']
            distance = np.sqrt((point_lat - mean_lat)**2 + (point_long - mean_long)**2)
            if distance <= 1000:
                map.plot(point_lat, point_long, '.', color=colors[int(region)], markersize=4)



def plot_stacked_histogram(ax, df):
    # Group by venueId and time_of_day, and sum the check-in counts for each group
    df_grouped = df.groupby([ 'time_of_day', 'regions'])['checkin_count'].sum().reset_index()

    df_pivot = df_grouped.pivot_table(values='checkin_count', index='time_of_day', columns='regions')

    df_pivot.plot(kind='bar', stacked=True, color=colors, ax=ax, alpha=0.5)

    ax.set_xticklabels(['00:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
)

    # set the x-axis label
    ax.set_xlabel('Time of day')

    # set the y-axis label
    ax.set_ylabel('Checkin count')

    # set the title
    ax.set_title('Checkin count by region and time of day for 1 week period')


# # replace 'path/to/file.csv' with the path to your CSV file
df = pd.read_csv('./csv/checkins.csv')



# Apply the function to the location column to extract the latitude and longitude values
df['longitude'], df['latitude'] = zip(*df['location'].apply(extract_coords))
kmeans = KMeans(n_clusters=8, random_state=0).fit(df[['latitude', 'longitude']])

## Elbow graph
# ssd = []
# for k in range(1, 16):
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(df[['latitude', 'longitude']])
#     ssd.append(kmeans.inertia_)

# #Plot the elbow graph
# plt.plot(range(1, 16), ssd, 'bx-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of squared distances')
# plt.title('Elbow graph for check-in clustering')
# plt.show()

# # Add the cluster labels to the data frame
df['regions'] = kmeans.labels_

# Print the resulting data frame
# Convert the datetime string to a datetime object
df['datetime'] = pd.to_datetime(df['hour'])

# Extract the day of the week (Monday = 0, Sunday = 6)
df['day_of_week'] = df['datetime'].dt.dayofweek

# Extract the time of day (morning = 0, afternoon = 1, evening = 2, night = 3)
df['time_of_day'] = pd.cut(df['datetime'].dt.hour, bins=list(range(25)), labels=False, include_lowest=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']



# Initialize the subplot function with 2 rows and 2 columns
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Remove the unused subplot

# Adjust the last plot to span both columns

print_city_map(ax1)

plot_regions_on_map(df, ax1)

ax1.set_title('Clustered check-ins regions')

plot_stacked_histogram(ax2,df)

# show the plot
plt.show()