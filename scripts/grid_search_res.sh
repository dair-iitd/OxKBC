#!/bin/sh

python3 grid_search_res.py $1

cat "$1"/results.csv | xclip -selection clipboard
