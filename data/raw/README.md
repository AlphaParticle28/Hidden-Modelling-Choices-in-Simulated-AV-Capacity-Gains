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
| `d04_text_station_5min_2026_09_25.txt.gz` | Fri 25 Sep 2026 (test day: never used for fitting) |
| `d04_text_station_5min_2026_09_29.txt.gz` | Tue 29 Sep 2026 (test day: never used for fitting) |

For detector positions, ramps and lane counts, also download one District 4 **Station Metadata**
file (in the Clearinghouse, choose the "Station Metadata" type) into this folder, for example
`d04_text_meta_2026_04_08.txt`. A file from a nearby date is fine, because detector locations rarely
change. The scripts use the newest one present. Without it, the bottleneck check falls back to open
sensor coordinates and has no ramp or lane information.

Leave the files gzipped; the scripts read them directly. The pipeline also accepts any other
`dXX_text_station_5min_YYYY_MM_DD.txt.gz` file placed here.

Each row is one station for one 5-minute interval. The first 12 columns are station-level fields,
followed by 5 columns per lane (samples, flow, occupancy, speed, observed flag). This project uses
lane 1 only.
