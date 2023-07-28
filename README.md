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

## PFHub

CabanaPF currently implements the PFHub 1a benchmark, available [here](https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/).  We also implement an alternative benchmark with periodic initial conditions.

## License

CabanaPF is distributed under an [open source 3-clause BSD license](LICENSE).

## Help

If you have questions regarding CabanaPF, please contact one of the developers:
David Joy (dhj0005@auburn.edu)

Sam Reeve (reevest@ornl.gov)

Steve DeWitt (dewittsj@ornl.gov)

## Acknowledgments

This work was supported in part by the U.S. Department
of Energy, Office of Science, Office of Workforce Development
for Teachers and Scientists (WDTS) under the Science
Undergraduate Laboratory Internships (SULI) program. 

This research was also supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative. 
