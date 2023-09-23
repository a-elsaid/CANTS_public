# Contineous Ant-based Neural Topolgy Search
## Abstract
>Ant-based Topology Search (ANTS) is a novel nature-inspired neural architecture search
(NAS) that is based on ant colony optimization (ACO). The neural structure search space is indirectly-encode in a massively connected structure, on which optimization agents (ants) swarm, traversing from inputs to outputs through the structure in a search for an optimum neural topology. Continuous Ant-based Topology Search (CANTS) is an advancement for ANTS that replaces the discrete search space with a 3D unbounded continuous one. Synthetic ant agents explore CANTS’ continuous search space based on the density and distribution of pheromones, strongly inspired by how ants move in the real world. This continuous search space allows CANTS to automate the design of artificial neural networks (ANNs) of any size, removing a fundamental limitation inherent to many current NAS algorithms that must operate within structures of a size that the user predetermines. CANTS also utilizes a fourth dimension in its search space, representing potential neural synaptic weights, transforming the solution from NAS to NeuroEvolution (NE). Adding this extra dimension allows CANTS agents to optimize both the architecture as well as the weights of an ANN without applying backpropagation (BP), which leads to a significant reduction in the time consumed in the optimization process

## Publications:
- [Backpropagation-free 4D continuous ant-based neural topology search (doi.org/10.1016/j.asoc.2023.110737)](https://doi.org/10.1016/j.asoc.2023.110737)
- [Optimizing long short-term memory recurrent neural networks using ant colony optimization to predict turbine engine vibration (doi.org/10.1016/j.asoc.2018.09.013)](https://doi.org/10.1016/j.asoc.2018.09.013)
- [Continuous Ant-Based Neural Topology Search (doi.org/10.1007/978-3-030-72699-7_19)](
https://doi.org/10.1007/978-3-030-72699-7_19)
- [Using ant colony optimization to optimize long short-term memory recurrent neural networks (doi.org/10.1145/3205455.3205637)](https://doi.org/10.1145/3205455.3205637)

## Note
This package works on time-series (tabular) data. Data samples are provided in the data folder. 


## Running The Program
- The program is testing on Linux-based operating systems. 
- Bash files are offered to run the different of optimization:
    - ANTS optimization: `run_single_ants_colony.sh`
    - CANTS optimization without Backpropagation: `run_single_cants_colony.sh`
    - CANTS optimzation with Backpropagation: `run_single_cants_colony_bp.sh`
    - Unoptimized training: `run_unoptimized.sh`
    - Use `bash` to run one of those files. e.g.:
    >```bash run_single_cants_colony.sh```

    - ### The commandline parameters:
        - `mpirun -n`: `n` is the number of CPUs to be used. Should be `n`$\le 2$ 
        - `--oversubscribe`: to assign more CPUs than is physically available.
        - `--data_dir`: data root directory 
        - `--data_files`: data files to be used for training and testing. Files should be seperated with `,`
        - `--input_names`: features used for input
        - `--output_names`: features used for output
        - `--living_time`: optimization iterations
        - `--out_dir`: directory to be used to save output files
        - `--log_dir`: directory to be used to save logging files
        - `--term_log_level`: level of logging displayed in terminal:
            - `DEBUG`: for more details
            - `INFO`: for less details
            - `WARNING`: for warning logs only or higher
            - `ERROR`: for error logs only or higher
        - `--log_file_name `: log file name. Each MPI worker process will have its own log file. The number of worker process will be attached at the end of the provided file name
        - `--file_log_leve`: levle of logging saved in log files
        - `--num_ants`: number of ants used in the optimization
        - `--loss_fun`: used loss function [options: `mse`: mean squared error, `mae`: mean absolute error]
        - `--act_fun`: activation funtion used [options: `sigmoid`, `tanh`, `relu`]

## Requirements 
- Required Python modules are listed in './src/requirements.txt'. They can be installed using 'pip install -r requirements.txt'
- This implementation utilizes Message Passing Interface (MPI) for parallel processing. OpenMPI should be installed before using running the code. 