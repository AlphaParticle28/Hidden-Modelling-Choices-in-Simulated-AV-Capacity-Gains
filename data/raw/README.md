# Raw data

The raw files are not stored in the repository (about 30 MB each). They are free to download from
the Caltrans PeMS Data Clearinghouse. A free account is needed.

1. Log in at https://pems.dot.ca.gov/
2. Open `https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=4&submit=Submit`
3. Under "Available Files", download these District 4 Station 5-Minute files into this folder:

| File | Day |
|---|---|
| `d04_text_station_5min_2026_08_27.txt.gz` | Thu 27 Aug 2026 |
| `d04_text_station_5min_2026_09_15.txt.gz` | Tue 15 Sep 2026 |
| `d04_text_station_5min_2026_09_16.txt.gz` | Wed 16 Sep 2026 |
| `d04_text_station_5min_2026_09_17.txt.gz` | Thu 17 Sep 2026 |
| `d04_text_station_5min_2026_09_22.txt.gz` | Tue 22 Sep 2026 |

Leave the files gzipped; the scripts read them directly. The pipeline also accepts any other
`dXX_text_station_5min_YYYY_MM_DD.txt.gz` file placed here.

Each row is one station for one 5-minute interval. The first 12 columns are station-level fields,
followed by 5 columns per lane (samples, flow, occupancy, speed, observed flag). This project uses
lane 1 only.
