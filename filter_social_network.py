import pandas as pd
from db_query_to_pd import db_query
from save_to_csv import save_to_csv

def filter_social_network():
    ## Make query to database
    participant_data = db_query("SELECT participantId, GROUP_CONCAT(DISTINCT jobId SEPARATOR ',') as jobIds, COUNT(DISTINCT jobId) as job_count, SUBSTRING_INDEX(GROUP_CONCAT(DISTINCT apartmentId ORDER BY apartmentId ASC SEPARATOR ','), ',', 1) as apartmentId, COUNT(DISTINCT apartmentId) as apartment_count FROM participant_data WHERE apartmentId != 0 GROUP BY participantId")

    ### Write to csv
    save_to_csv('participant_data.csv', participant_data)