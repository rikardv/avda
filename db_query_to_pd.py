from connect_db import connect_to_db
from save_to_csv import save_to_csv
import pandas as pd

def db_query(query):
    # Connect to the MySQL database
    cnx = connect_to_db()

    # Create a cursor object to execute queries
    cursor = cnx.cursor()

    # Execute the query
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchall()

    # Convert the results into a pandas data frame
    df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
        
    # Close the cursor and connection
    cursor.close()
    cnx.close()

    return df
