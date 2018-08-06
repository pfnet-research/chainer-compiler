#!/usr/bin/env bash

set -eu

clang-format -i \
             common/*.{h,cc} \
             compiler/*.{h,cc} \
             runtime/*.{h,cc} \
             tools/*.cc
