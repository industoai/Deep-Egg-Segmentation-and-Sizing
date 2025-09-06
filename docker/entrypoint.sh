#!/bin/bash -l
set -e
if [ "$#" -eq 0 ]; then
  exec egg_segmentation_size --help
else
  exec "$@"
fi
