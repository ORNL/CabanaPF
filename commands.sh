#!/bin/bash

CABANA_DIR="/home/kokkos/src/CabanaPF"
BUILD_DIR="$CABANA_DIR/build"

#these are provided for convience for rapid development and debugging
function b() {
    cd $BUILD_DIR
    cmake $CABANA_DIR
    make PFHub1a
    make Test
}
function r() {
    /home/kokkos/src/CabanaPF/build/examples/PFHub1a
}
function t() {
    /home/kokkos/src/CabanaPF/build/unit_test/Test
}

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
    echo "Created build directory"
fi
