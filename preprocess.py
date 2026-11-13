import csv

# 6 crop types in the dataset
CROPS = ["Irrigated Palay", "Rainfed Palay", "Palay", "White Corn", "Yellow Corn", "Corn"]


def load_data(filepath="dataset.csv"):
    with open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # row 2 has years, row 3 has quarters
    years_row = rows[2]
    quarters_row = rows[3]

    time_columns = []
    current_year = None

    for c in range(2, len(years_row)):
        year_text = years_row[c].strip()
        if year_text != "":
            current_year = int(year_text)

        quarter_text = quarters_row[c].strip()
        quarter_num = None
        if "Quarter 1" in quarter_text:
            quarter_num = 1
        elif "Quarter 2" in quarter_text:
            quarter_num = 2
        elif "Quarter 3" in quarter_text:
            quarter_num = 3
        elif "Quarter 4" in quarter_text:
            quarter_num = 4

        # skip anything past 2025
        if current_year is not None and current_year <= 2025:
            time_columns.append((current_year, quarter_num))

    # each crop block is 109 rows, starting at these row positions
    crop_starts = [
        (4, "Irrigated Palay"),
        (113, "Rainfed Palay"),
        (222, "Palay"),
        (331, "White Corn"),
        (440, "Yellow Corn"),
        (549, "Corn")
    ]

    records = []
    current_region = "National"

    for crop_info in crop_starts:
        start_row = crop_info[0]
        crop_type = crop_info[1]

        for r in range(start_row, start_row + 109):
            if r >= len(rows):
                break

            raw_loc = rows[r][1].strip()
            if raw_loc == "":
                continue

            # dot prefix tells us if its a region, province, or city
            if raw_loc.startswith("PHILIPPINES"):
                geo_level = "National"
                clean_loc = "Philippines"
                current_region = "National"
            elif raw_loc.startswith("...."):
                clean_loc = raw_loc[4:].strip()
                if "City" in clean_loc:
                    geo_level = "City"
                else:
                    geo_level = "Province"
            elif raw_loc.startswith(".."):
                clean_loc = raw_loc[2:].strip()
                geo_level = "Region"
                current_region = clean_loc
            else:
                clean_loc = raw_loc
                geo_level = "Province"

            # psa adds footnote letters like a/, b/ — remove them
            for marker in [" a/", " b/", " c/", " d/", "d/"]:
                clean_loc = clean_loc.replace(marker, "")
                current_region = current_region.replace(marker, "")
            clean_loc = clean_loc.strip()
            current_region = current_region.strip()

            # missing cells show up as '..', '-', or blank
            values = []
            for c in range(2, 2 + len(time_columns)):
                if c < len(rows[r]):
                    cell = rows[r][c].strip()
                else:
                    cell = ""

                if cell == ".." or cell == "" or cell == "-":
                    values.append(None)
                else:
                    try:
                        clean_num = cell.replace(",", "")
                        values.append(float(clean_num))
                    except ValueError:
                        values.append(None)

            item = {
                "crop": crop_type,
                "location": clean_loc,
                "geo_level": geo_level,
                "region": current_region,
                "values": values
            }
            records.append(item)

    return time_columns, records


def average_of_all_years(records):
    results = []

    for record in records:
        if record["geo_level"] == "National":
            continue

        # add up all non-missing quarters
        total = 0.0
        count = 0
        for val in record["values"]:
            if val is not None:
                total = total + val
                count = count + 1

        # multiply by 4 to get yearly total from the quarterly average
        if count > 0:
            quarterly_average = total / count
            annual_average = round(quarterly_average * 4.0, 2)
        else:
            annual_average = None

        item = {
            "crop": record["crop"],
            "geo_level": record["geo_level"],
            "location": record["location"],
            "region": record["region"],
            "average_yearly": annual_average
        }
        results.append(item)

    return results


def total_per_year(time_columns, records):
    results = []

    for record in records:
        if record["geo_level"] == "National":
            continue

        sums = {}
        counts = {}

        # bucket each value into its year
        for index in range(len(time_columns)):
            year = time_columns[index][0]
            val = record["values"][index]

            if year not in sums:
                sums[year] = 0.0
                counts[year] = 0

            if val is not None:
                sums[year] = sums[year] + val
                counts[year] = counts[year] + 1

        # skip the year if any quarter is missing
        totals = {}
        for year in sums:
            if counts[year] == 4:
                totals[year] = round(sums[year], 2)
            else:
                totals[year] = None

        item = {
            "crop": record["crop"],
            "geo_level": record["geo_level"],
            "location": record["location"],
            "region": record["region"],
            "totals_per_year": totals
        }
        results.append(item)

    return results


