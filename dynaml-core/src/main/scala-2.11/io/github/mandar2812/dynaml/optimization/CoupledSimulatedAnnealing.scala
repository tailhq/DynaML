/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.optimization

/**
  * Implementation of the Coupled Simulated Annealing algorithm
  * for global optimization.
  *
  * @author mandar datum 25/6/15.
  *
  * */
class CoupledSimulatedAnnealing[M <: GloballyOptimizable](model: M) extends
  AbstractCSA[M, M](model: M) with
  GlobalOptimizer[M] {

  override def optimize(initialConfig: Map[String, Double],
                        options: Map[String, String] = Map()) = {


    val landscape = performCSA(initialConfig, options).toMap
    val optimum = landscape.keys.min

    logger.info(
      "Optimum value of energy is: "+optimum+
      " at: \n"+GlobalOptimizer.prettyPrint(landscape(optimum)))

    //Persist the current configuration to the model memory
    if(options.contains("persist") && (options("persist") == "true" || options("persist") == "1"))
      system.persist(landscape(optimum))

    (system, landscape(optimum))
  }
}


