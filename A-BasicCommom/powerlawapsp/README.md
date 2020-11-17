# Supplementary Documents For Lower Bounds on Rate of Convergence of Matrix Products in All-Pairs Shortest Path of Social Network

# 1. Dataset:
A dataset for Chinese Movie Actors Sina Weibo Following Relationship 2011-2015.

* `dataset` folder.  
    - actorId.txt  
    This file contains profiles of movie actors who has an account in Sina Weibo. Following information are provided for each actor: the number of fans(10k., Sina Weibo verified name, Sina Weibo Account Id, Sina Weibo Nickname, actor self description. All actors have at least 10,000 of fans, it shows that all users in this dataset are famous to some point.  

* `data` folder.
    - train.npz  
    This file is a compressed adjacent NumPy matrix of 8508 actors. The adjacent matrix `m` stores connection for each node pairs. For all node `u`, set `m[u,u]=0`, and for any node `u` follows node `v`, then set `m[u, v]=1`, other values in matrix `m` are set to `m[u, v]=3.4028235e+38` which is the maximum value of floating-point in 32bit.

    - apsp.npy
    This file is extract from train.npz.
 
# 2. Library Requirement  

## 2.1 Programming Environment
Installation of python 3.6+, NVIDIA CUDA10.1, JDK 1.8.0, Maven 3.6.1 is required.

## 2.2 Python Dependant 

See requirements.txt

## 2.3 Java Libraries

Java libraries need to be installed, install dependencies using command: `mvn install:install-file -DpomFile=pom[-gpu].xml`.  

The installed full list is as follows:  
1. org.nd4j:nd4j-cuda-10.0-platform:1.0.0-beta6  
2. org.nd4j:nd4j-native-platform:1.0.0-beta6  
3. org.nd4j:nd4j-native:windows-x86_64-avx2:1.0.0-beta6  
4. org.nd4j:nd4j-native:linux-x86_64-avx2:1.0.0-beta6  
5. org.slf4j:slf4j-api:1.7.25  
6. ch.qos.logback:logback-classic:1.2.3  
7. org.projectlombok:lombok:1.18.10  
8. org.springframework.boot:spring-boot-starter-web:2.2.5.RELEASE  
9. org.springframework.boot:spring-boot-starter-test:2.2.5.RELEASE  


## 2.4 Java jar build.

Use maven to build executing jar. Using command: `mvn package -f pom[-gpu].xml`,
 then get the result jar file `apsp-cpu.jar` or `apsp-gpu.jar`.

# 3. Experimental Guideline

## 3.1 Floyd-Warshal

Command parameters explained as follows:  
1. input matrix in numpy format  
2. diameter of the network  
3. output matrix  
4. algorithm name  

* floydwarshall for Floyd-Warshal all pairs shortest path algorithm
  
* powerlawbound for this paper's algorithm.  
   
Full command as follows:  
`java -jar apsp-cpu.jar matrix.npy 8508 apsp.npy floydwarshall`

## 3.2 Alon N

* CPU implementation

`docker run --rm --gpus=all -it powerlawapsp:1.0 python3 train.py -c config/dpmm_config.json`

* GPU implementation

`docker run --rm --gpus=all -it powerlawapsp:1.0 python3 train.py -c config/dpmm_gpu_config.json`


## 3.3 PowerLawBound

Build the dockers

* pytorch:1.1 docker file:
  See `Dockerfile-pytorch_1.1`
* pwoerlawapsp:1.0 docker file:
  See `Dockerfile`

An example of execute command:  
`docker run --rm --gpus=all -it powerlawapsp:1.0 python3 train.py -c config/{config}.json`

### 3.3.1 PowerLawBound-CPU-NumPy
Use config file `naive_apsp_config.json`

### 3.3.2 PowerLawBound-GPU-CuPy
Use config file `naive_apsp_gpu_config.json`

### 3.3.3 PowerLawBound-CPU-SciPy-sparse-Numpy
Use config file `apsp_config.json`

### 3.3.4 PowerLawBound-GPU-CuPy-cuSparse-Cupy
Use config file `apsp_gpu_config.json`

### 3.3.5 PowerLawBound-GPU-CUBLAS
`java -jar apsp-gpu.jar matrix.npy 8 apsp.npy`

### 3.3.6 PowerLawBound-CPU-OPENBLAS
`java -jar apsp-cpu.jar matrix.npy 8 apsp.npy`

