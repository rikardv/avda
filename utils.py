import re
import numpy as np
from matplotlib.patches import Polygon
import pandas as pd

def extract_coords(point_str):
    # Define a regular expression pattern to match the latitude and longitude values
    pattern = r'POINT \((-?\d+\.\d+)\s+(-?\d+\.\d+)\)'
    match = re.match(pattern, point_str)
    if match:
        latitude = float(match.group(2))
        longitude = float(match.group(1))
        return latitude, longitude
    else:
        return None, None


def print_city_map(ax):
    data = pd.read_csv("./csv/Buildings.csv", sep=",")
    for i in range(len(data)):

        #Process string
        type = data["buildingType"][i]
        test = (data["location"][i])
        test = test.strip("POLYGON")
        test = test.replace("(","")
        test = test.replace(")","")
        test = test.strip(" ")
        test = test.strip("))").split(", ")
        polygon = []

        #Create tuples
        for tuple in test:
            tuple.replace("(","")
            xy = tuple.split(" ")
            polygon.append([xy[0],xy[1]])
        y = np.array(polygon)

        #color according to building type
        if type =="Commercial":
            color = "bisque"
        elif type =="Residental":
            color = "slategray"

        p = Polygon(y, facecolor = color, edgecolor= "k", linewidth=0.1,)
        ax.add_patch(p)

    #Fair enough
    ax.set_xlim([-5000,2500])
    ax.set_ylim([0,8000])
    # set the title
    ax.set_title('test')

    return ax