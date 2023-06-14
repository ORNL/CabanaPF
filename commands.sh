#!/bin/bash

CABANA_DIR="/home/kokkos/src/CabanaPF"
BUILD_DIR="$CABANA_DIR/build"

#these are provided for convience for rapid development and debugging
function b() {
    cd $BUILD_DIR
    cmake $CABANA_DIR
    make CabanaPF
    make Test
}
function r() {
    cd $BUILD_DIR
    ./CabanaPF
}
function t() {
    cd $BUILD_DIR
    ./Test
}

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
    echo "Created build directory"
fi
