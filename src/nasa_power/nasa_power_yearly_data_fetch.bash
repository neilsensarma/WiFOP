#!/bin/bash

mkdir -p power_ca_2020_2023 && cd power_ca_2020_2023

base='https://power.larc.nasa.gov/api/temporal/daily/regional'
latmin=32.5; latmax=42.0
tiles=("-124.5 -119.5" "-119.5 -114.1")   # each tile ≤10°, ≥2°
vars=(T2M RH2M WS10M PRECTOT PS)
years=(2020 2021 2022 2023)
months=(06 07 08 09)  # Jun–Sep

# helper: last day of month (non-leap/leap not needed here, but good to have)
last_day () {
  local y=$1 m=$2
  case "$m" in
    01|03|05|07|08|10|12) echo 31 ;;
    04|06|09|11) echo 30 ;;
    02)
      # simple leap-year check
      if (( (y % 400 == 0) || (y % 4 == 0 && y % 100 != 0) )); then echo 29; else echo 28; fi
      ;;
  esac
}

for y in "${years[@]}"; do
  for m in "${months[@]}"; do
    sd="${y}${m}01"
    ed="${y}${m}$(last_day "$y" "$m")"

    common="community=AG&latitude-min=$latmin&latitude-max=$latmax&start=$sd&end=$ed&time-standard=UTC&format=CSV"

    for v in "${vars[@]}"; do
      i=1
      for tile in "${tiles[@]}"; do
        read lonmin lonmax <<<"$tile"
        url="$base?parameters=$v&longitude-min=$lonmin&longitude-max=$lonmax&$common"
        out="${v}_${y}${m}_tile${i}.csv"
        echo "Fetching $v $y-$m tile $i ..."
        curl -L --fail --retry 3 --retry-delay 2 --compressed "$url" -o "$out"
        ((i++))
        sleep 0.5
      done
    done
  done
done
