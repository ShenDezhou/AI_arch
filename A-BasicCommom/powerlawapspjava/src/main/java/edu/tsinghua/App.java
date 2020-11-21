//Copyright (c) 2020-2040 Dezhou Shen, Tsinghua University
//        Licensed under the Apache License, Version 2.0 (the "License");
//        you may not use this file except in compliance with the License.
//        You may obtain a copy of the License at
//        http://www.apache.org/licenses/LICENSE-2.0
//        Unless required by applicable law or agreed to in writing, software
//        distributed under the License is distributed on an "AS IS" BASIS,
//        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//        See the License for the specific language governing permissions and
//        limitations under the License.
package edu.tsinghua;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;


public class App {

    public static void main(String[] args) {

        String adjacent_numpy_matrix = "matrix.npy";
        int diameter = 9;
        String algorithm = "mmdp";
        if (args.length > 0)
            adjacent_numpy_matrix = args[0];
        if (args.length > 1)
            diameter = Integer.parseInt(args[1]);
        if (args.length > 2)
            algorithm = args[2];

        INDArray apsp = null;
        log.info("epsilon=" + EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.epsilon);
        log.info("diameter of graph:"+ diameter);
        log.info("algorithm:"+ algorithm);
        try {
            Instant b = Instant.now();
            INDArray adjacencyNumpy = Nd4j.createFromNpyFile(new File(adjacent_numpy_matrix));
            log.info("number of rows:"+ adjacencyNumpy.shape()[0]);
            log.info("matrix loaded;");
            long startTime = System.currentTimeMillis();
            log.info("algorithm:"+ algorithm);
            if (algorithm.equals("floydwarshall")) {
                apsp = FloydWarshall.allPairsShortestPath(adjacencyNumpy, adjacencyNumpy);
            } else {
                apsp = EarlyStopRepeatedSquareMatrixMultiplicationDistanceProduct.allPairsShortestPath(adjacencyNumpy, diameter);
            }
            log.info("Total execution time: " + (System.currentTimeMillis() - startTime) + "ms");
            log.info("distance product calculation finished;");
            Nd4j.writeAsNumpy(apsp, new File("result_" + adjacent_numpy_matrix));
            log.info("write to IO finished.");
            long exectime = Duration.between(b, Instant.now()).toMillis();
            log.info(exectime);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
