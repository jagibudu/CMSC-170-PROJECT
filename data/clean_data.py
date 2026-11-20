from data.preprocess import load_data, average_of_all_years

# only the 4 base crops, "Palay" and "Corn" are just sums of these
FEATURE_CROPS = ["Irrigated Palay", "Rainfed Palay", "White Corn", "Yellow Corn"]

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
            # old row up to 2023 Q4 + NIR row from 2024 Q1 = full 144 quarters, 136 is the index of 2024 Q1
            # this also replaces the stray 0.0 in the old row's 2025 Q1
            new_values = record["values"][:136] + nir_rows[key]["values"][136:]
            record = dict(record)
            record["values"] = new_values
        joined.append(record)
    return joined

# It returns the cleaned records
# add more cleaning functions here if yall find other data issues
def clean_records(records):
    records = keep_places(records)
    records = drop_empty(records)
    records = join_negros(records)
    return records


# copied from hierarchical.py file
# Prepare province-level feature matrix
def prepare_feature_matrix(averages):
    province_data = {}

    for record in averages:
        if record["geo_level"] != "Province":
            continue

        crop = record["crop"]

        if crop not in FEATURE_CROPS:
            continue

        province = record["location"]
        region = record["region"]
        value = record["average_yearly"]

        key = (region, province)

        if key not in province_data:
            province_data[key] = {}

        province_data[key][crop] = value

    province_names = []
    X = []
    excluded = []

    for key, crops in province_data.items():
        if any(
            crop not in crops or crops[crop] is None
            for crop in FEATURE_CROPS
        ):
            excluded.append(key)
            continue

        province_names.append(key)

        row = [
            crops[crop]
            for crop in FEATURE_CROPS
        ]

        X.append(row)

    return province_names, X, excluded

# Standardize data using Z-score
def standardize_data(X):
    if not X:
        raise ValueError("Feature matrix is empty.")

    num_rows = len(X)
    num_features = len(X[0])

    means = []
    stds = []

    for j in range(num_features):
        mean = sum(X[i][j] for i in range(num_rows)) / num_rows

        variance = sum(
            (X[i][j] - mean) ** 2
            for i in range(num_rows)
        ) / num_rows

        std = variance ** 0.5

        means.append(mean)
        stds.append(std)

    standardized = []

    for row in X:
        new_row = []

        for j in range(num_features):
            if stds[j] == 0:
                new_row.append(0.0)
            else:
                new_value = (row[j] - means[j]) / stds[j]
                new_row.append(new_value)

        standardized.append(new_row)

    return standardized, means, stds