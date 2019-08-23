#!/bin/bash

BASE_DIR="/users4/kliao/data/New_York_Times_Annotated_Corpus"

for year in "$BASE_DIR"/*
do
    cd "$year"
    pwd
    for tar in "$year"/*.tgz
    do
        echo "$tar"
        tar -zxf "$tar"
    done
done
