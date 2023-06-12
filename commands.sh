#!/bin/bash

CABANA_DIR="/home/kokkos/src/CabanaPF"
BUILD_DIR="$CABANA_DIR/build"

function b() {
    cd $BUILD_DIR
    cmake $CABANA_DIR
    make CabanaPF
}
function r() {
    cd $BUILD_DIR
    ./CabanaPF
}

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
    echo "Created build directory"
fi
