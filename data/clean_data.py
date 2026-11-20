import numpy as np
from preprocess import load_data, average_of_all_years



# keep only the province or city, removes the national and region rows
def keep_places(records):
    places = []
    for record in records:
        if record["geo_level"] == "Province" or record["geo_level"] == "City":
            places.append(record)
    return places
