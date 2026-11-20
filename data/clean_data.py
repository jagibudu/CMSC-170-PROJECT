import numpy as np
from preprocess import load_data, average_of_all_years

# keep only the province or city, removes the national and region rows
def keep_places(records):
    places = []
    for record in records:
        if record["geo_level"] == "Province" or record["geo_level"] == "City":
            places.append(record)
    return places

# Removes records where all values are None, meaning there is no data for that record.
def drop_empty(records):
    with_data = []
    for record in records:
        # check if the record has at least one real value
        has_value = False
        for val in record["values"]:
            if val is not None:
                has_value = True
                break
        if has_value:
            with_data.append(record)
    return with_data

# joins Negros Occidental, Negros Oriental and Siquijor back into one series each
# without this, each of the 3 provinces shows up twice (Region VI / VII row and NIR row)
# old row has 1990-2023, NIR row has 2024-2025
# see the PSA notes at the end of dataset.csv (a/ and b/: part of Region VI / VII prior to 2024)
# This function is written with help of Claude sonnet 5.5
def join_negros(records):
    # find the NIR rows first
    nir_rows = {}
    for record in records:
        if record["region"] == "NEGROS ISLAND REGION":
            nir_rows[(record["crop"], record["location"])] = record

    joined = []
    for record in records:
        # skip NIR rows so each province is only counted once
        # their 2024-2025 values get copied into the old rows below
        if record["region"] == "NEGROS ISLAND REGION":
            continue
        # only the old Negros rows have a matching NIR row, the rest pass unchanged
        key = (record["crop"], record["location"])
        if key in nir_rows:
             # old row up to 2023 Q4 + NIR row from 2024 Q1 = full 144 quarters
            # this also replaces the stray 0.0 in the old row's 2025 Q1
            new_values = record["values"][:136] + nir_rows[key]["values"][136:]
            record = dict(record)
            record["values"] = new_values
        joined.append(record)
    return joined

