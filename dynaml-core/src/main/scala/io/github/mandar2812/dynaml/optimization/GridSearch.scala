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
  * @author mandar2812 datum 24/6/15.
  *
  * An implementation of Grid Search
  * global optimization for Kernel Models
  */
class GridSearch[M <: GloballyOptimizable](model: M)
    extends AbstractGridSearch[M, M](model)
    with GlobalOptimizer[M] {

  override def optimize(
    initialConfig: Map[String, Double],
    options: Map[String, String] = Map()
  ) = {

    println(
      "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-"
    )
    println(
      "Grid Search "
    )
    println(
      "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-"
    )
    println()

    val energyLandscape =
      getEnergyLandscape(initialConfig, options, meanFieldPrior).toMap
    val optimum = energyLandscape.keys.min

    print("Optimum value of energy is: ")
    pprint.pprintln(optimum)
    println("Configuration: ")
    pprint.pprintln(energyLandscape(optimum))

    //Persist the current configuration to the model memory
    if (options.contains("persist") && (options("persist") == "true" || options(
          "persist"
        ) == "1"))
      system.persist(energyLandscape(optimum))

    (system, energyLandscape(optimum))
  }
}
