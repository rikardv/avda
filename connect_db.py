import mysql.connector
import config

def connect_to_db():
    # Set up connection parameters
    setup = {
        'user': config.user,
        'password': config.password,
        'host': config.host,
        'database': config.database,
        'raise_on_warnings': True
    }

    # Establish a connection to the MySQL database
    cnx = mysql.connector.connect(**setup)

    return cnx