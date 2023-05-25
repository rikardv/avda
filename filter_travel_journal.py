import pandas as pd
from db_query_to_pd import db_query
from save_to_csv import save_to_csv
import uuid
import pickle


## Make query to database
## filter out 1 week of travel journals

def generate_pickled_travel_journal(id, purpose, name):
    result_dict = {}
    query = ""
    if purpose:
        query = f"SELECT * FROM `travel_journal` WHERE travelStartTime >= '2022-03-01T05:20:00Z' AND travelStartTime <= '2022-03-06T12:20:00Z' and purpose = '{purpose}'"
    elif id:
        query = f"SELECT * FROM `travel_journal` WHERE travelStartTime >= '2022-03-01T05:20:00Z' AND travelStartTime <= '2022-03-06T12:20:00Z' and participantId = {id}"
    data = db_query(query)

    print('found nr of rows: ', len(data))

    df_participant_data = pd.read_csv("./csv/ParticipantStatusLogs1.csv") 

    for index, row in data.iterrows():
        hash_id = str(uuid.uuid4())
        id = row['participantId']
        start = row['travelStartTime']
        end = row['travelEndTime']
        stamps = df_participant_data.loc[(df_participant_data['participantId'] == id) & (df_participant_data['timestamp'] >= start) & (df_participant_data['timestamp'] <= end)]
        result_dict[hash_id] = stamps
        # q: how can i print the index of to total number of rows
        # a: use the index of the for loop and print it out with the total number of rows       
        print('completed ', index, ' of ', len(data)) 

    if purpose:
        # Store the result_dict in a pickle file
        with open(f"{name}_journeys.pickle", "wb") as file:
            pickle.dump(result_dict, file)
    elif id:
        # Store the result_dict in a pickle file
        with open(f"{id}_journeys.pickle", "wb") as file:
            pickle.dump(result_dict, file)