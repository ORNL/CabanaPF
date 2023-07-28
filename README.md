# CabanaPF

Phase-field with the Cabana particle library

## Dependencies

CabanaPF has the following dependencies:

|Dependency | Version  | Required | Details|
|---------- | -------  |--------  |------- |
|CMake      | 3.20+    | Yes      | Build system
|Cabana     | master   | Yes      | Performance portable particle algorithms
|GTest      | 1.10+    | Yes      | Unit test framework

Cabana must be built with the following in order to work with CabanaPF:
|Cabana Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|MPI        | GPU-Aware if CUDA/HIP enabled | Yes | Message Passing Interface
|Kokkos     | 3.6.0+  | Yes      | Performance portable on-node parallelism
|heFFTe 	| 2.1.0   | Yes      | (Experimental) Performance portable fast Fourier transforms

## License

CabanaPF is distributed under an [open source 3-clause BSD license](LICENSE).
