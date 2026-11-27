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

## 5. Clustering Methods

We implemented three clustering algorithms from scratch to compare how different approaches group Philippine provinces based on their Palay and Corn production characteristics.

### 5.1 K-Means Clustering

K-Means is a partition-based clustering algorithm that groups data points according to their proximity to cluster centroids. Each province is assigned to the cluster with the nearest centroid using Euclidean distance. The centroids are then updated by calculating the mean of the features of all provinces assigned to each cluster. This process repeats until the assignments or centroids converge.

### 5.2 Hierarchical Clustering

We implemented agglomerative hierarchical clustering using average linkage. Initially, each province is treated as an individual cluster. The algorithm repeatedly merges the two clusters with the smallest average pairwise distance until the desired number of clusters is reached.

Average linkage calculates the distance between two clusters based on the average distance between all pairs of data points from the two clusters.

### 5.3 Gaussian Mixture Model (GMM)

Gaussian Mixture Model is a probabilistic clustering algorithm that represents the data as a mixture of Gaussian distributions. Unlike K-Means, GMM assigns probabilities indicating how likely a province belongs to each cluster.

Our implementation uses K-Means to initialize the GMM parameters, including the initial means, weights, and covariance matrices. It then performs the Expectation-Maximization (EM) algorithm to iteratively estimate the responsibilities and update the parameters until convergence or the maximum number of iterations is reached. The final cluster assignment is based on the component with the highest responsibility.

---

## 6. Experimental Setup

To ensure a fair comparison, all three clustering algorithms used the same dataset, features, and standardized input.

| Parameter | Description |
|---|---|
| Dataset | Philippine Palay and Corn Production Data |
| Coverage | 1990–2025 |
| Samples | 82 provinces |
| Features | 4 crop production features |
| Features used | Irrigated Palay, Rainfed Palay, White Corn, Yellow Corn |
| Preprocessing | Annual averaging and Z-score standardization |
| Number of clusters (K) | 3 |
| Distance measure | Euclidean distance |
| Hierarchical linkage | Average linkage |
| K-Means random seed | 10 |
| GMM initialization | K-Means initialization |
| GMM maximum iterations | 100 |
| GMM tolerance | 0.000001 |

The four features represent the annual average production of each crop in tons. Production values were averaged over the selected period from 1990 to 2025 to represent the general production characteristics of each province.

The data were standardized using Z-score standardization before clustering. This helps prevent differences in the magnitude of crop production values from disproportionately influencing the clustering process.

Since the project uses unsupervised learning, no train-test split was performed. The algorithms were applied to the same standardized dataset to compare their resulting cluster structures.

---

## 7. Evaluation Metrics

To evaluate and compare the quality of the clusters produced by the three algorithms, we implemented two internal evaluation metrics: Silhouette Score and Calinski-Harabasz Index.

### 7.1 Silhouette Score

The Silhouette Score measures how well each data point fits within its assigned cluster compared to other clusters.

For each sample:

- **a(i)** is the average distance between the sample and other points within its own cluster.
- **b(i)** is the lowest average distance between the sample and points belonging to another cluster.

The silhouette value is calculated as:

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

The overall Silhouette Score is the average of the silhouette values of all samples.

The score ranges from -1 to 1:

- A value close to **1** indicates that the sample is well matched to its cluster.
- A value close to **0** indicates that the sample is near a cluster boundary.
- A value close to **-1** indicates that the sample may be assigned to the wrong cluster.

A higher Silhouette Score generally indicates better-defined and more separated clusters.

### 7.2 Calinski-Harabasz Index

The Calinski-Harabasz (CH) Index evaluates clustering quality by comparing the separation between clusters with the compactness within clusters.

It is calculated using the ratio of between-cluster dispersion to within-cluster dispersion, adjusted according to the number of clusters and samples.

A higher Calinski-Harabasz Index indicates that clusters are relatively compact and well separated from one another.

### 7.3 Why We Used Both Metrics

We used both metrics because they evaluate clustering quality from different perspectives.

The Silhouette Score measures how well individual samples fit within their clusters compared to neighboring clusters, while the Calinski-Harabasz Index evaluates the overall separation and compactness of the clusters.

Using both provides a more balanced evaluation instead of relying on only one metric to determine clustering performance.

---

## 8. Results

The three clustering algorithms generated different groupings of provinces based on their Palay and Corn production characteristics.

The resulting clusters generally reflect differences in production characteristics, including provinces with major Palay production, provinces with major Corn production, and provinces with relatively smaller or moderate production levels.

The results were examined using cluster visualizations, crop production profiles, cluster distributions, and internal evaluation metrics.

### 8.1 K-Means Results

K-Means produced clusters based on the proximity of provinces to their respective centroids. Its resulting groupings showed production similarities among provinces based on the four standardized crop features.

### 8.2 Hierarchical Clustering Results

Hierarchical clustering generated groups through successive merging of the closest clusters using average linkage. Its results provided a different grouping structure compared with K-Means and GMM.

### 8.3 Gaussian Mixture Model Results

GMM generated clusters based on probabilistic membership to Gaussian components. Its results reflect the probability-based grouping of provinces according to their production characteristics.

---

## 9. Comparison of Clustering Methods

The clustering algorithms were compared using the Silhouette Score and Calinski-Harabasz Index.

Based on the evaluation results:

- **K-Means** obtained the highest Calinski-Harabasz Index, indicating stronger cluster compactness and separation based on this metric.
- **Hierarchical Clustering** obtained the highest Silhouette Score, indicating better-defined clusters based on the average relationship between within-cluster and nearest-other-cluster distances.
- **GMM** obtained an intermediate Calinski-Harabasz result but the lowest Silhouette Score among the three methods.

Overall, K-Means was considered the strongest method based on the Calinski-Harabasz Index, while its Silhouette Score was relatively close to that of Hierarchical Clustering.

These findings demonstrate that different clustering algorithms can produce different results depending on how they define and form clusters. No single metric completely captures clustering quality, which is why both evaluation measures were considered.

---

## 10. Discussion and Limitations

The clustering results demonstrate that Philippine provinces can be grouped according to similarities in their historical Palay and Corn production characteristics. The three algorithms provide different perspectives on these similarities due to their distinct clustering mechanisms.

However, the study has several limitations.

First, averaging production data across the selected years reduces the dataset to general production characteristics. This may hide seasonal patterns, fluctuations, growth, decline, and changes in production over time.

Second, the study only considers four crop features: Irrigated Palay, Rainfed Palay, White Corn, and Yellow Corn. Other agricultural factors, such as land area, weather conditions, soil characteristics, and farming practices, are not included.

Third, the project uses internal clustering metrics. Although Silhouette Score and Calinski-Harabasz Index help evaluate cluster structure, they do not establish that the resulting groups represent all real-world agricultural conditions or that the groupings directly explain agricultural productivity.

Therefore, the results should be interpreted as an exploratory analysis of production similarities rather than definitive classifications of agricultural provinces.

---

## 11. Conclusion

This project explored the clustering of Philippine provinces based on their historical Palay and Corn production data from 1990 to 2025. Using four crop production features and three clustering algorithms—K-Means, Hierarchical Clustering, and Gaussian Mixture Model—we examined how provinces can be grouped according to similarities in their production characteristics.

The results showed that the clustering methods produced different groupings and performed differently based on the selected evaluation metrics. K-Means obtained the highest Calinski-Harabasz Index, while Hierarchical Clustering achieved the highest Silhouette Score.

Through this project, we demonstrated the implementation and comparison of clustering algorithms from scratch and highlighted the importance of preprocessing, feature selection, and evaluation metrics in unsupervised learning.

---

## 12. How to Run

### Requirements

- Python 3.x
- NumPy
- Matplotlib (if running visualizations)

### Setup

Clone the repository:

```bash
git clone https://github.com/jagibudu/CMSC-170-PROJECT.git
```

Navigate to the project directory:

```bash
cd CMSC-170-PROJECT
```

Install the required dependency:

```bash
pip install numpy
```

Ensure that the dataset is located in the `data/` directory and named `dataset.csv`.

> **Note:** Run commands depend on the project's actual entry-point files. Use the appropriate main or runner script to execute the clustering algorithms and evaluation. Refer to the project's source files for the corresponding execution functions.

---

## 13. Reproducing Results

To reproduce the clustering results, the same dataset and preprocessing pipeline must be used across all three algorithms.

The preprocessing pipeline performs the following steps:

1. Load the Philippine Palay and Corn production dataset.
2. Select the four crop production features.
3. Process the records at the province level.
4. Handle regional changes and combine Negros-related records where necessary.
5. Exclude provinces with incomplete feature data.
6. Calculate annual average production values for the selected period.
7. Standardize the features using Z-score standardization.
8. Apply the clustering algorithms using three clusters (K = 3).
9. Evaluate the resulting clusters using Silhouette Score and Calinski-Harabasz Index.

For reproducibility, the K-Means initialization uses a random seed of 10. GMM is initialized using K-Means and uses a maximum of 100 iterations with a convergence tolerance of 0.000001.

All clustering algorithms use the same standardized feature matrix to ensure a consistent basis for comparison.

---

## 14. Team Contributions

| Member | Contribution |
|---|---|
| **Jason Gian B. Duran** | Data preprocessing, plotting and visualization, and PowerPoint presentation |
| **Kevin Aaron A. Salazar** | Project pages, K-Means implementation, and Calinski-Harabasz Index |
| **Ron Kirby M. Arches** | Milestone reporting and Gaussian Mixture Model implementation |
| **Carlos James R. Miranda** | Hierarchical Clustering implementation, Silhouette Score, and README documentation |

Although tasks were divided among members, the group collaboratively participated in brainstorming, troubleshooting, discussions, evaluation, and the completion of the project deliverables.

---

## 15. References and AI Disclosure

### Dataset Source

Philippine Statistics Authority (PSA) – OpenStat

The dataset contains historical Philippine Palay and Corn production records used for the project's data preprocessing and clustering analysis.

### AI Tool Disclosure

AI tools, including Claude, were used as assistance during development for code troubleshooting, explanations, and selected programming-related tasks.

All generated suggestions and code were reviewed, tested, modified, and integrated by the group. The group is responsible for understanding the implemented algorithms, code, experimental procedures, evaluation, and reported findings.
