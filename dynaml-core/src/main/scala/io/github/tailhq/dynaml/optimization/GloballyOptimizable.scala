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
package io.github.tailhq.dynaml.optimization

import io.github.tailhq.dynaml.pipes._
import io.github.tailhq.dynaml.models._

/**
  * A common binding characteristic between all "globally optimizable"
  * models i.e. models where hyper-parameters can
  * be optimized/tuned.
  *
  * @author tailhq, date 23/6/15.
  *
  * */
trait GloballyOptimizable {

  /**
   * Stores the names of the hyper-parameters
   * */
  protected var hyper_parameters: List[String]

  /**
   * A Map which stores the current state of
   * the system.
   * */
  protected var current_state: Map[String, Double]

  def _current_state = current_state

  def _hyper_parameters = hyper_parameters

  /**
   * Calculates the energy of the configuration,
   * in most global optimization algorithms
   * we aim to find an approximate value of
   * the hyper-parameters such that this function
   * is minimized.
   *
   * @param h The value of the hyper-parameters in the configuration space
   * @param options Optional parameters about configuration
   * @return Configuration Energy E(h)
   * */
  def energy(h: Map[String, Double],
             options: Map[String, String] = Map()): Double

  def persist(state: Map[String, Double]): Unit = {}

}

object GloballyOptimizable {

  /**
    * Instantiate a custom implementation of systems/models with
    * hyper-parameters and associated fitness/energy.
    * 
    * */
  def apply[T, M <: Model[T, _, _]](
    system: M, hyp_params: List[String],
    starting_state: Map[String, Double],
    energyFunction: DataPipe2[M, Map[String, Double], Double]): GloballyOptimizable =
    new GloballyOptimizable {

      override protected var  hyper_parameters = hyp_params

      override protected var current_state = starting_state

      override def energy(
        h: Map[String, Double],
        options: Map[String, String] = Map()) = energyFunction(system, h)


    }

}

trait GloballyOptWithGrad extends GloballyOptimizable {

  /**
    * Calculates the gradient energy of the configuration and
    * subtracts this from the current value of h to yield a new
    * hyper-parameter configuration.
    *
    * Over ride this function if you aim to implement a gradient based
    * hyper-parameter optimization routine like ML-II
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @return Gradient of the objective function as a Map
    * */
  def gradEnergy(h: Map[String, Double])
  : Map[String, Double] = Map()
}
