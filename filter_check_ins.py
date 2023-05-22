import pandas as pd
from db_query_to_pd import db_query
from save_to_csv import save_to_csv

## Make query to database
query = "SELECT check_in_journal.participantId, check_in_journal.timestamp, check_in_journal.venueId, check_in_journal.venueType, pubs.pubId, pubs.hourlyCost, pubs.maxOccupancy, COALESCE(pubs.location, restaurants.location) AS location, pubs.buildingId, restaurants.restaurantId, restaurants.foodCost, restaurants.maxOccupancy, pubs.buildingId FROM check_in_journal LEFT JOIN pubs ON check_in_journal.venueId = pubs.pubId LEFT JOIN restaurants ON check_in_journal.venueId = restaurants.restaurantId WHERE timestamp >= '2022-03-01' AND timestamp < '2022-03-06' AND (venueType = 'Pub' OR venueType = 'Restaurant')"
data = db_query(query)


## Perform some more preprocessing of the data
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.floor('H')
data = data.groupby(['hour', 'venueId', 'location', 'venueType']).size().reset_index(name='checkin_count')

### Write to csv
save_to_csv('checkins.csv', data)