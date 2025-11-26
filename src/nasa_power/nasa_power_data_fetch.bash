#!/bin/bash

mkdir -p power_ca_jul_sep2023 && cd power_ca_jul_sep2023

base='https://power.larc.nasa.gov/api/temporal/daily/regional'
latmin=32.5; latmax=42.0
start=20230701; end=20230930
common="community=AG&latitude-min=$latmin&latitude-max=$latmax&start=$start&end=$end&time-standard=UTC&format=CSV"

vars=(T2M RH2M WS10M PRECTOT PS)

tiles=("-124.5 -119.5" "-119.5 -114.1")

for v in "${vars[@]}"; do
  i=1
  for tile in "${tiles[@]}"; do
    read lonmin lonmax <<<"$tile"
    url="$base?parameters=$v&longitude-min=$lonmin&longitude-max=$lonmax&$common"
    curl -L "$url" -o "${v}_tile${i}.csv"
    ((i++))
    sleep 0.5
  done
done
