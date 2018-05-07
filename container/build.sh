#!/usr/bin/env bash

docker build -t isi/typer .
docker run --rm -v $(pwd)/..:/semantic-labeling -it isi/typer ./build_ext.sh
