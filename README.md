# CMSC 170 Project: Palay and Corn Production Clustering (1990–2025)

**Group Name:** Mga Alipin ng 170  
**Members:**
- Kevin Aaron A. Salazar
- Jason Gian B. Duran
- Ron Kirby M. Arches
- Carlos James R. Miranda

**GitHub Repository:** [https://github.com/jagibudu/CMSC-170-PROJECT](https://github.com/jagibudu/CMSC-170-PROJECT)

---

## 1. Project Overview & Rules Compliance

This repository contains the data preparation and preprocessing pipeline for Philippine historical Palay (Rice) and Corn production data (1990–2025), sourced from the Philippine Statistics Authority (PSA).

### Dataset

**Dataset:** Palay and Corn: Volume of Production in Metric Tons by Ecosystem/Croptype, Quarter, Semester, Region and Province, 1987–2026 from the PSA Statistical Database.

**Source:** [PSA Statistical Database](https://openstat.psa.gov.ph/)

### Goal

We will group Philippine regions and provinces based on their historical palay and corn production profiles. The goal is to discover natural groupings of agricultural productivity and uncover underlying patterns in seasonal yields across different geographical areas in the Philippines.

### Proposed Methods

We will be exclusively using clustering algorithms to ensure that the different methods solve the exact same unsupervised learning task, guaranteeing a fair and reproducible comparison. Applying only clustering techniques to the dataset allows us to evaluate their performance using the same standard metrics.

| Algorithm | Description |
| :--- | :--- |
| **K-Means Clustering** | Partitions locations into *k* clusters by minimizing within-cluster variance |
| **Hierarchical Clustering** | Builds a tree of nested clusters (dendrogram) without requiring a preset *k* |
| **Gaussian Mixture Model (GMM)** | Models clusters as overlapping Gaussian distributions for soft assignments |

### Rules Compliance

- **No forbidden high-level libraries:** Pandas and Scikit-learn are not used.
- **Pure Standard Library:** Implemented using standard Python (`csv` only).
- **Target Timeframe:** Uses historical data strictly from **1990 to 2025** (144 quarters).

---

## 2. Exportable Python Functions (`preprocess.py`)

All functions in `preprocess.py` can be directly imported into clustering algorithm codes.

### How to Import:
```python
from preprocess import load_data, average_of_all_years, total_per_year, CROPS
```

---

### Function 1: `load_data(filepath="dataset.csv")`

Reads and loads the raw dataset into clean Python structures up to year 2025.

#### Return Value:
Returns a tuple: `(time_columns, records)`

1. **`time_columns`**: list of 144 `(year, quarter)` integer tuples:
   ```python
   [
       (1990, 1), (1990, 2), (1990, 3), (1990, 4),
       ...,
       (2025, 1), (2025, 2), (2025, 3), (2025, 4)
   ]
   ```
2. **`records`**: list of 654 dictionaries representing each location and crop series:
   ```python
   {
       'crop': 'Palay',
       'location': 'Abra',
       'geo_level': 'Province',   # 'Region', 'Province', 'City', or 'National'
       'region': 'CORDILLERA ADMINISTRATIVE REGION (CAR)',
       'values': [7270.0, 7756.0, 5776.0, ..., 26244.8]  # length 144 (float or None)
   }
   ```

---

### Function 2: `average_of_all_years(records)`

Calculates the average yearly (annual) production across all 36 years (1990–2025) for each location and crop.

#### Return Value:
Returns a list of dictionaries (648 entries):
```python
[
    {
        'crop': 'Palay',
        'geo_level': 'Province',
        'location': 'Abra',
        'region': 'CORDILLERA ADMINISTRATIVE REGION (CAR)',
        'average_yearly': 57149.25    # Average yearly production in Metric Tons (MT)
    },
    ...
]
```



---

### Function 3: `total_per_year(time_columns, records)`

Calculates the annual total production (sum of 4 quarters) for every year (1990 to 2025) for each location and crop.

#### Return Value:
Returns a list of dictionaries (648 entries):
```python
[
    {
        'crop': 'Palay',
        'geo_level': 'Province',
        'location': 'Abra',
        'region': 'CORDILLERA ADMINISTRATIVE REGION (CAR)',
        'totals_per_year': {
            1990: 25637.0,
            1991: 24109.0,
            1992: 20483.0,
            ...,
            2025: 73327.6
        }
    },
    ...
]
```


---

## 3. Results: Average Production Across All Years (1990–2025)

Values represent average yearly volume of production in **Metric Tons (MT)**.

### A. Regional Aggregates (All 17 Regions)

| Region | Irrigated Palay | Rainfed Palay | Total Palay | White Corn | Yellow Corn | Total Corn |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM)** | 172,366 | 342,480 | 514,846 | 477,383 | 299,968 | 777,351 |
| **CORDILLERA ADMINISTRATIVE REGION (CAR)** | 285,117 | 40,910 | 326,026 | 17,347 | 123,312 | 140,659 |
| **MIMAROPA REGION** | 596,680 | 298,294 | 894,974 | 8,901 | 80,027 | 88,928 |
| **NEGROS ISLAND REGION** | 387,062 | 124,597 | 511,660 | 80,752 | 25,401 | 106,153 |
| **REGION I (ILOCOS REGION)** | 1,011,093 | 442,445 | 1,453,539 | 39,995 | 299,418 | 339,412 |
| **REGION II (CAGAYAN VALLEY)** | 1,849,506 | 167,767 | 2,017,273 | 51,967 | 1,218,090 | 1,270,056 |
| **REGION III (CENTRAL LUZON)** | 2,471,855 | 271,314 | 2,743,169 | 19,825 | 151,213 | 171,038 |
| **REGION IV-A (CALABARZON)** | 317,518 | 78,350 | 395,868 | 25,585 | 40,617 | 66,202 |
| **REGION IX (ZAMBOANGA PENINSULA)** | 343,361 | 187,758 | 531,120 | 174,098 | 22,135 | 196,233 |
| **REGION V (BICOL REGION)** | 701,217 | 280,724 | 981,942 | 42,791 | 137,149 | 179,940 |
| **REGION VI (WESTERN VISAYAS)** | 970,830 | 829,000 | 1,799,829 | 51,819 | 154,221 | 206,040 |
| **REGION VII (CENTRAL VISAYAS)** | 159,865 | 89,398 | 249,262 | 150,024 | 8,217 | 158,242 |
| **REGION VIII (EASTERN VISAYAS)** | 432,650 | 302,220 | 734,871 | 73,539 | 5,932 | 79,470 |
| **REGION X (NORTHERN MINDANAO)** | 532,254 | 56,955 | 589,209 | 397,805 | 648,507 | 1,046,312 |
| **REGION XI (DAVAO REGION)** | 389,442 | 43,297 | 432,739 | 183,992 | 45,448 | 229,440 |
| **REGION XII (SOCCSKSARGEN)** | 885,327 | 186,298 | 1,071,625 | 379,508 | 781,216 | 1,160,724 |
| **REGION XIII (CARAGA)** | 267,843 | 135,632 | 403,475 | 60,156 | 38,165 | 98,322 |

---

### B. Specific Locations (Provinces and Cities)

| Province / City | Parent Region | Irrigated Palay | Rainfed Palay | Total Palay | White Corn | Yellow Corn | Total Corn |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Abra** | CORDILLERA ADMINISTRATIVE REGION (CAR) | 40,435 | 16,714 | 57,149 | 11,044 | 472 | 11,516 |
| **Agusan del Norte** | REGION XIII (CARAGA) | 58,214 | 19,085 | 77,298 | 12,433 | 4,048 | 16,480 |
| **Agusan del Sur** | REGION XIII (CARAGA) | 103,476 | 77,262 | 180,738 | 39,402 | 31,466 | 70,869 |
| **Aklan** | REGION VI (WESTERN VISAYAS) | 69,118 | 45,463 | 114,581 | 325 | 1,197 | 1,523 |
| **Albay** | REGION V (BICOL REGION) | 146,803 | 27,350 | 174,153 | 2,796 | 41,534 | 44,330 |
| **Antique** | REGION VI (WESTERN VISAYAS) | 150,009 | 72,474 | 222,483 | 526 | 4,475 | 5,001 |
| **Apayao** | CORDILLERA ADMINISTRATIVE REGION (CAR) | 51,020 | 15,305 | 66,325 | 792 | 17,524 | 18,316 |
| **Aurora** | REGION III (CENTRAL LUZON) | 73,921 | 6,238 | 80,159 | 693 | 10,046 | 10,739 |
| **Bacolod City** ¹ | NEGROS ISLAND REGION | - | - | - | - | - | - |
| **Basilan** | BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM) | 1,905 | 1,296 | 3,201 | 1,437 | 0 | 1,437 |
| **Bataan** | REGION III (CENTRAL LUZON) | 109,565 | 1,098 | 110,663 | 934 | 3,280 | 4,214 |
| **Batanes** | REGION II (CAGAYAN VALLEY) | 1 | 25 | 25 | 30 | 84 | 114 |
| **Batangas** | REGION IV-A (CALABARZON) | 38,187 | 14,063 | 52,250 | 14,733 | 7,898 | 22,631 |
| **Benguet** | CORDILLERA ADMINISTRATIVE REGION (CAR) | 13,187 | 1,632 | 14,819 | 87 | 103 | 190 |
| **Biliran** | REGION VIII (EASTERN VISAYAS) | 50,699 | 1,080 | 51,778 | 388 | 206 | 594 |
| **Bohol** | REGION VII (CENTRAL VISAYAS) | 92,594 | 79,487 | 172,081 | 14,075 | 1,327 | 15,401 |
| **Bukidnon** | REGION X (NORTHERN MINDANAO) | 311,206 | 28,944 | 340,150 | 109,935 | 583,575 | 693,509 |
| **Bulacan** | REGION III (CENTRAL LUZON) | 248,713 | 62,908 | 311,621 | 1,714 | 468 | 2,182 |
| **Butuan City** ¹ | REGION XIII (CARAGA) | - | - | - | - | - | - |
| **Cagayan** | REGION II (CAGAYAN VALLEY) | 549,983 | 106,197 | 656,180 | 15,868 | 276,650 | 292,518 |
| **Camarines Norte** | REGION V (BICOL REGION) | 49,695 | 20,548 | 70,243 | 1,344 | 404 | 1,748 |
| **Camarines Sur** | REGION V (BICOL REGION) | 378,909 | 110,036 | 488,945 | 8,587 | 93,500 | 102,087 |
| **Camiguin** | REGION X (NORTHERN MINDANAO) | 2,086 | 4 | 2,090 | 480 | 69 | 548 |
| **Capiz** | REGION VI (WESTERN VISAYAS) | 87,913 | 203,483 | 291,396 | 603 | 38,495 | 39,098 |
| **Catanduanes** | REGION V (BICOL REGION) | 13,428 | 14,732 | 28,161 | 236 | 172 | 408 |
| **Cavite** | REGION IV-A (CALABARZON) | 37,746 | 3,757 | 41,503 | 1,426 | 1,521 | 2,948 |
| **Cebu** | REGION VII (CENTRAL VISAYAS) | 11,860 | 1,120 | 12,979 | 77,013 | 4,388 | 81,401 |
| **City of Davao** | REGION XI (DAVAO REGION) | 11,867 | 8,409 | 20,276 | 13,375 | 4,515 | 17,889 |
| **Cotabato (North Cotabato)** | REGION XII (SOCCSKSARGEN) | 326,266 | 90,818 | 417,085 | 122,824 | 235,873 | 358,698 |
| **Davao Occidental** | REGION XI (DAVAO REGION) | 342 | 108 | 449 | 7,343 | 1,358 | 8,701 |
| **Davao Oriental** | REGION XI (DAVAO REGION) | 50,778 | 8,886 | 59,664 | 47,985 | 4,280 | 52,265 |
| **Davao de Oro (Compostela Valley)** | REGION XI (DAVAO REGION) | 63,917 | 10,195 | 74,112 | 25,873 | 14,238 | 40,112 |
| **Davao del Norte** | REGION XI (DAVAO REGION) | 142,640 | 12,704 | 155,344 | 34,943 | 9,623 | 44,566 |
| **Davao del Sur** | REGION XI (DAVAO REGION) | 119,898 | 2,995 | 122,893 | 54,473 | 11,434 | 65,907 |
| **Dinagat Islands** | REGION XIII (CARAGA) | 2,127 | 954 | 3,082 | 69 | 33 | 101 |
| **Eastern Samar** | REGION VIII (EASTERN VISAYAS) | 9,677 | 31,965 | 41,642 | 161 | 117 | 278 |
| **Guimaras** | REGION VI (WESTERN VISAYAS) | 10,787 | 30,768 | 41,555 | 528 | 469 | 998 |
| **Ifugao** | CORDILLERA ADMINISTRATIVE REGION (CAR) | 45,452 | 1,448 | 46,900 | 1,997 | 54,909 | 56,906 |
| **Ilocos Norte** | REGION I (ILOCOS REGION) | 211,255 | 38,478 | 249,733 | 10,422 | 35,833 | 46,255 |
| **Ilocos Sur** | REGION I (ILOCOS REGION) | 99,717 | 74,311 | 174,027 | 6,601 | 49,118 | 55,719 |
| **Iloilo** | REGION VI (WESTERN VISAYAS) | 389,791 | 376,930 | 766,721 | 13,092 | 72,619 | 85,712 |
| **Isabela** | REGION II (CAGAYAN VALLEY) | 1,026,271 | 48,631 | 1,074,902 | 31,091 | 786,538 | 817,629 |
| **Kalinga** | CORDILLERA ADMINISTRATIVE REGION (CAR) | 120,232 | 3,554 | 123,786 | 1,106 | 35,621 | 36,726 |
| **La Union** | REGION I (ILOCOS REGION) | 83,979 | 49,255 | 133,234 | 7,029 | 13,762 | 20,791 |
| **Laguna** | REGION IV-A (CALABARZON) | 122,158 | 2,441 | 124,599 | 2,305 | 1,212 | 3,517 |
| **Lanao del Norte** | REGION X (NORTHERN MINDANAO) | 134,929 | 24,167 | 159,096 | 207,583 | 26,925 | 234,508 |
| **Lanao del Sur** | BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM) | 47,873 | 102,027 | 149,900 | 236,111 | 98,454 | 334,565 |
| **Leyte** | REGION VIII (EASTERN VISAYAS) | 288,039 | 106,224 | 394,262 | 49,366 | 3,229 | 52,595 |
| **Maguindanao del Norte** | BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM) | 81,516 | 190,140 | 271,656 | 76,590 | 244,082 | 320,671 |
| **Maguindanao del Sur** | BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM) | 114,280 | 221,817 | 336,098 | 231,995 | 183,136 | 415,131 |
| **Marinduque** | MIMAROPA REGION | 8,846 | 14,195 | 23,041 | 406 | 360 | 765 |
| **Masbate** | REGION V (BICOL REGION) | 23,978 | 81,926 | 105,905 | 29,610 | 1,189 | 30,799 |
| **Misamis Occidental** | REGION X (NORTHERN MINDANAO) | 62,231 | 2,727 | 64,958 | 33,048 | 2,397 | 35,445 |
| **Misamis Oriental** | REGION X (NORTHERN MINDANAO) | 21,802 | 1,112 | 22,914 | 46,759 | 35,542 | 82,301 |
| **Mountain Province** | CORDILLERA ADMINISTRATIVE REGION (CAR) | 14,791 | 2,257 | 17,048 | 2,322 | 14,683 | 17,005 |
| **Negros Occidental** | NEGROS ISLAND REGION | 318,662 | 107,853 | 426,515 | 37,377 | 23,532 | 60,909 |
| **Negros Occidental** | REGION VI (WESTERN VISAYAS) | 276,660 | 104,986 | 381,647 | 38,906 | 39,139 | 78,045 |
| **Negros Oriental** | NEGROS ISLAND REGION | 66,912 | 16,568 | 83,480 | 42,601 | 1,724 | 44,324 |
| **Negros Oriental** | REGION VII (CENTRAL VISAYAS) | 56,272 | 9,041 | 65,312 | 57,019 | 2,562 | 59,581 |
| **Northern Samar** | REGION VIII (EASTERN VISAYAS) | 10,180 | 69,242 | 79,421 | 9,681 | 1,637 | 11,318 |
| **Nueva Ecija** | REGION III (CENTRAL LUZON) | 1,216,218 | 118,769 | 1,334,987 | 8,962 | 11,713 | 20,675 |
| **Nueva Vizcaya** | REGION II (CAGAYAN VALLEY) | 206,993 | 6,891 | 213,885 | 2,978 | 50,598 | 53,576 |
| **Occidental Mindoro** | MIMAROPA REGION | 189,091 | 93,938 | 283,029 | 4,333 | 48,708 | 53,041 |
| **Oriental Mindoro** | MIMAROPA REGION | 246,379 | 62,476 | 308,855 | 200 | 3,119 | 3,319 |
| **Palawan** | MIMAROPA REGION | 135,059 | 113,328 | 248,387 | 3,240 | 27,253 | 30,493 |
| **Pampanga** | REGION III (CENTRAL LUZON) | 318,277 | 9,104 | 327,380 | 6,037 | 35,269 | 41,305 |
| **Pangasinan** | REGION I (ILOCOS REGION) | 616,143 | 280,402 | 896,545 | 15,942 | 200,705 | 216,647 |
| **Puerto Princesa City** ¹ | MIMAROPA REGION | - | - | - | - | - | - |
| **Quezon** | REGION IV-A (CALABARZON) | 94,621 | 55,435 | 150,056 | 6,674 | 29,512 | 36,187 |
| **Quirino** | REGION II (CAGAYAN VALLEY) | 66,259 | 6,023 | 72,282 | 2,000 | 104,220 | 106,220 |
| **Rizal** | REGION IV-A (CALABARZON) | 24,806 | 2,654 | 27,460 | 446 | 473 | 920 |
| **Romblon** | MIMAROPA REGION | 17,305 | 14,358 | 31,662 | 723 | 587 | 1,310 |
| **Samar** | REGION VIII (EASTERN VISAYAS) | 12,942 | 86,097 | 99,039 | 8,672 | 467 | 9,139 |
| **Sarangani** | REGION XII (SOCCSKSARGEN) | 26,771 | 8,423 | 35,195 | 59,877 | 54,954 | 114,831 |
| **Siquijor** | NEGROS ISLAND REGION | 1,488 | 177 | 1,665 | 774 | 146 | 920 |
| **Siquijor** | REGION VII (CENTRAL VISAYAS) | 1,970 | 199 | 2,170 | 5,385 | 88 | 5,473 |
| **Sorsogon** | REGION V (BICOL REGION) | 88,404 | 26,132 | 114,535 | 217 | 350 | 568 |
| **South Cotabato** | REGION XII (SOCCSKSARGEN) | 234,648 | 36,819 | 271,467 | 150,574 | 344,966 | 495,540 |
| **Southern Leyte** | REGION VIII (EASTERN VISAYAS) | 61,114 | 7,614 | 68,728 | 5,271 | 274 | 5,545 |
| **Sultan Kudarat** | REGION XII (SOCCSKSARGEN) | 297,641 | 50,238 | 347,879 | 46,233 | 145,423 | 191,655 |
| **Sulu** | BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM) | 85 | 4,179 | 4,264 | 2,085 | 0 | 2,085 |
| **Sulu** ¹ | REGION IX (ZAMBOANGA PENINSULA) | - | - | - | - | - | - |
| **Surigao del Norte** | REGION XIII (CARAGA) | 48,836 | 14,867 | 63,703 | 597 | 453 | 1,049 |
| **Surigao del Sur** | REGION XIII (CARAGA) | 55,191 | 23,463 | 78,654 | 7,656 | 2,166 | 9,822 |
| **Tarlac** | REGION III (CENTRAL LUZON) | 428,858 | 45,812 | 474,669 | 1,256 | 89,652 | 90,908 |
| **Tawi-tawi** | BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM) | 0 | 1,600 | 1,600 | 962 | 0 | 962 |
| **Zambales** | REGION III (CENTRAL LUZON) | 76,304 | 27,386 | 103,690 | 230 | 786 | 1,016 |
| **Zamboanga City** | REGION IX (ZAMBOANGA PENINSULA) | 17,316 | 5,963 | 23,279 | 4,717 | 3,151 | 7,868 |
| **Zamboanga Sibugay** | REGION IX (ZAMBOANGA PENINSULA) | 55,026 | 54,664 | 109,690 | 13,599 | 1,616 | 15,216 |
| **Zamboanga del Norte** | REGION IX (ZAMBOANGA PENINSULA) | 47,236 | 41,693 | 88,930 | 51,097 | 2,826 | 53,923 |
| **Zamboanga del Sur** | REGION IX (ZAMBOANGA PENINSULA) | 223,783 | 85,438 | 309,221 | 104,685 | 14,541 | 119,226 |

> ¹ No production data recorded in the PSA dataset for 1990–2025. These locations are listed in the source file but all quarterly values are missing (`None`). They are **not** zero-production areas.

---

## 4. Results: Sample Annual Total Production (Palay, Selected Years)

Values represent annual total production (sum of 4 quarters) in **Metric Tons (MT)**.

| Region | 1990 | 2000 | 2010 | 2020 | 2025 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **BARMM** | 211,341 | 347,549 | 623,343 | 725,093 | 944,951 |
| **CAR** | 165,585 | 255,820 | 400,415 | 376,473 | 339,250 |
| **MIMAROPA** | 483,109 | 758,719 | 857,532 | 1,183,149 | 1,314,220 |
| **NEGROS ISLAND REGION** | - | - | - | - | 523,622 |
| **REGION I** | 847,691 | 1,207,890 | 1,558,373 | 1,902,662 | 1,963,770 |
| **REGION II** | 1,164,758 | 1,785,137 | 1,745,722 | 2,645,777 | 3,014,907 |
| **REGION III** | 1,963,919 | 1,960,232 | 2,958,415 | 3,635,148 | 3,541,191 |
| **REGION IV-A** | 421,188 | 376,439 | 390,189 | 386,920 | 384,110 |
| **REGION IX** | 342,690 | 443,598 | 552,645 | 645,407 | 752,889 |
| **REGION V** | 686,998 | 672,293 | 1,081,078 | 1,294,991 | 1,230,731 |
| **REGION VI** | 886,732 | 1,608,337 | 1,789,693 | 2,295,580 | 1,485,758 |
| **REGION VII** | 182,940 | 215,192 | 270,449 | 260,524 | 248,604 |
| **REGION VIII** | 337,459 | 517,558 | 964,145 | 927,095 | 803,050 |
| **REGION X** | 365,544 | 499,605 | 586,442 | 781,825 | 835,446 |
| **REGION XI** | 515,745 | 388,183 | 402,811 | 466,764 | 523,456 |
| **REGION XII** | 517,660 | 1,044,794 | 1,185,196 | 1,264,117 | 1,204,510 |
| **CARAGA** | 225,917 | 308,066 | 405,871 | 503,331 | 563,925 |

---

