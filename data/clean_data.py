# fixes(ai assisted) for psa dataset

CROPS = ["Irrigated Palay", "Rainfed Palay", "Palay", "White Corn", "Yellow Corn", "Corn"]

# provinces that moved to a new region in 2024. psa kept the old row (stops after 2023,
# then shows fake 0.0 values)
MOVED_PROVINCES = [
    ("Negros Occidental", "REGION VI (WESTERN VISAYAS)", "NEGROS ISLAND REGION", 2024),
    ("Negros Oriental", "REGION VII (CENTRAL VISAYAS)", "NEGROS ISLAND REGION", 2024),
    ("Siquijor", "REGION VII (CENTRAL VISAYAS)", "NEGROS ISLAND REGION", 2024)
]
# rows with no usable data
REMOVED_ROWS = [
    ("Sulu", "REGION IX (ZAMBOANGA PENINSULA)"),
    ("Bacolod City", "NEGROS ISLAND REGION"),
    ("Butuan City", "REGION XIII (CARAGA)"),
    ("Puerto Princesa City", "MIMAROPA REGION"),
    ("Batanes", "REGION II (CAGAYAN VALLEY)")
]
 
# we only use these 4, since "Palay" and "Corn" are just their sums (would count twice)
BASE_CROPS = ["Irrigated Palay", "Rainfed Palay", "White Corn", "Yellow Corn"]
 
 
def clean_records(time_columns, records):
    # copy only the province and city rows
    table = {}
    for record in records:
        if record["geo_level"] == "Province" or record["geo_level"] == "City":
            key = (record["crop"], record["location"], record["region"])
            table[key] = {
                "crop": record["crop"],
                "location": record["location"],
                "region": record["region"],
                "values": list(record["values"])   # a copy, so the original records stay unchanged
            }
 
    # remove the rows with no data that is listed in REMOVED_ROWS.
    for crop in CROPS:
        for removed in REMOVED_ROWS:
            location = removed[0]
            region = removed[1]
            del table[(crop, location, region)]
 
    # join the two rows of each moved province into one
    # the old row has the data up to 2023, the new row has the data from 2024 on.
    # we copy the new row's 2024+ values into the old row, then delete the new row
    for moved in MOVED_PROVINCES:
        location = moved[0]
        old_region = moved[1]
        new_region = moved[2]
        switch_year = moved[3]
 
        # position of the switch year's first quarter in the values list
        start = time_columns.index((switch_year, 1))
 
        for crop in CROPS:
            old_values = table[(crop, location, old_region)]["values"]
            new_values = table[(crop, location, new_region)]["values"]
 
            # replace the old row's values from the switch year on
            for index in range(start, len(time_columns)):
                old_values[index] = new_values[index]
 
            del table[(crop, location, new_region)]
 
    # join maguindanao del sur and del norte into one "Maguindanao"
    # del norte only exists from 2024 (blank before that), its harvest was inside del sur until then
    region = "BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM)"
    for crop in CROPS:
        sur_values = table[(crop, "Maguindanao del Sur", region)]["values"]
        norte_values = table[(crop, "Maguindanao del Norte", region)]["values"]
 
        # add the two rows quarter by quarter, a blank del norte quarter counts as 0
        joined_values = []
        for index in range(len(time_columns)):
            if norte_values[index] is None:
                joined_values.append(sur_values[index])
            else:
                joined_values.append(sur_values[index] + norte_values[index])
 
        # replace the two rows with the joined one
        del table[(crop, "Maguindanao del Sur", region)]
        del table[(crop, "Maguindanao del Norte", region)]
        table[(crop, "Maguindanao", region)] = {
            "crop": crop,
            "location": "Maguindanao",
            "region": region,
            "values": joined_values
        }
 
    # return the cleaned rows as a list, same format as load_data's records
    clean = list(table.values())
    return clean
 
 
def build_features(time_columns, clean):

    # group the rows by area
    # areas["Nueva Ecija"] = {"region": ..., "Irrigated Palay": [144 values], "Rainfed Palay": [...], ...}
    areas = {}
    for record in clean:
        location = record["location"]
        if location not in areas:
            areas[location] = {"region": record["region"]}
        areas[location][record["crop"]] = record["values"]
 
    # sorted so the rows always come out in the same order
    locations = sorted(areas)
   
    # used shares instead of tons so the clusters 
    # reflect when and what provinces harvest, not how much they produce.”
    # seasonal shares of each crop in each area
    # share of a quarter = (all harvests in that quarter, 1990-2025) / (all harvests, 1990-2025)
    shares = {}
    for crop in BASE_CROPS:
        share_sums = [0.0, 0.0, 0.0, 0.0]  
        grown_count = 0                     # number of areas that grow this crop
 
        for location in locations:
            values = areas[location][crop]
 
            # add up each quarter over all the years
            quarter_totals = [0.0, 0.0, 0.0, 0.0]
            for index in range(len(time_columns)):
                quarter = time_columns[index][1]   # 1, 2, 3 or 4
                quarter_totals[quarter - 1] = quarter_totals[quarter - 1] + values[index]
 
            crop_total = sum(quarter_totals)
 
            if crop_total == 0:
                # crop not grown here, a share would be 0 / 0, so it gets filled in below
                shares[(location, crop)] = None
            else:
                crop_shares = []
                for q in range(4):
                    share = quarter_totals[q] / crop_total
                    crop_shares.append(share)
                    share_sums[q] = share_sums[q] + share
                shares[(location, crop)] = crop_shares
                grown_count = grown_count + 1
 
        # areas that don't grow this crop get the average shares of the areas that do,
        # so "not grown" doesn't look like a strange harvest pattern
        average_shares = []
        for q in range(4):
            average_shares.append(share_sums[q] / grown_count)
 
        for location in locations:
            if shares[(location, crop)] is None:
                shares[(location, crop)] = average_shares
 
    #build the row of 19 features for each area
    results = []
    for location in locations:
        features = []
 
        # 16 seasonal shares: Q1 to Q4 of each of the 4 base crops
        for crop in BASE_CROPS:
            features = features + shares[(location, crop)]
 
        # total tons of each crop over all the years
        irrigated = sum(areas[location]["Irrigated Palay"])
        rainfed = sum(areas[location]["Rainfed Palay"])
        white = sum(areas[location]["White Corn"])
        yellow = sum(areas[location]["Yellow Corn"])
        palay = irrigated + rainfed
        corn = white + yellow
 
        # 3 crop mix features (the other side of each is 1 minus it, e.g. rainfed = 1 - irrigated)
        features.append(irrigated / palay)        # irrigated share of palay
        features.append(yellow / corn)            # yellow share of corn
        features.append(palay / (palay + corn))   # palay share of all output
 
        results.append({
            "location": location,
            "region": areas[location]["region"],
            "features": features
        })
 
    # step 4: one dictionary per area, with its 19 features in "features"
    return results
