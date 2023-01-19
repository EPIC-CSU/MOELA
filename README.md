# MOELA: Multi-Objective Evolutionary/Learning Design Space Exploration Framework

This repository contains the python algorithms for the MOELA and two comparison methods named MOEA/D[1] and MOOS[2]. MOELA is a multiple-objective optimization (MOO) framework, which can improve both the speed and quality of high dimensional design space exploration. 

S. Qi, Y. Li, S. Pasricha, and R.G. Kim, “MOELA: A Multi-Objective Evolutionary/Learning Design Space Exploration Framework for 3D Heterogeneous Manycore Platforms," Design, Automation & Test in Europe Conference & Exhibition (DATE), 2023

## Overview

1. This repository has three directories, which include the code for 3obj, 4obj, and 5obj design space problems respectively.
2. Inside each directory, there are python codes for MOELA, MOEA/D, and MOOS. Meanwhile, there are weight subdirectory and traffic subdirectory. The weight subdirectory contains weights used by MOELA and MOEA/D. And the traffic subdirectory contains the NoC traffic data obtained from the Rodinia Benchmark[3].
3. Use the shell scripts to launch different algorithms, which also contains the inputs (random seed and application) algorithms need. There are normal scripts to be launched on linux system like ubuntu. At the same time there are scripts to be launched on HPC system like Alpine in CSU.
4. After launching the scripts, algorithm results will be saved in the subdirectory. 

## Reference

[1] Q. Zhang and H. Li, “MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,” IEEE TEVC, vol. 11(6), 2007.\
[2] A. Deshwal, et al., “MOOS: A Multi-Objective Design Space Exploration and Optimization Framework for NoC Enabled Manycore Systems,” ACM TECS, vol. 18(5s), pp. 1-23, 2019.\
[3] S. Che, et al., “Rodinia: A benchmark suite for heterogeneous computing,” IISWC, pp. 44-54, 2009
