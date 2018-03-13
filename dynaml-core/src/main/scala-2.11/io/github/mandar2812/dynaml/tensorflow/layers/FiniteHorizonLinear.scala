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
package io.github.mandar2812.dynaml.tensorflow.layers

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, Regularizer}
import org.platanios.tensorflow.api.{---, Output, Shape, tf}

/**
  * Projection of a finite horizon multivariate
  * time series onto an observation space.
  *
  * @param units The degrees of freedom or dimensionality of the dynamical system
  * @param observables The dimensionality of the observations at each time epoch.
  * @author mandar2812 date 11/03/2018
  * */
case class FiniteHorizonLinear(
  override val name: String,
  units: Int, observables: Int, horizon: Int,
  weightsInitializer: Initializer = RandomNormalInitializer(),
  biasInitializer: Initializer = RandomNormalInitializer(),
  regularization: Regularizer = L2Regularizer()) extends
  Layer[Output, Output](name) {

  override val layerType: String = s"FHLinear[states:$units, horizon:$horizon, observables:$observables]"

  override protected def _forward(input: Output, mode: Mode): Output = {
    val weights      = tf.variable(
      s"$name/Weights", input.dataType, Shape(observables, units),
      weightsInitializer, regularizer = regularization)

    val bias         = tf.variable(
      s"$name/Bias", input.dataType, Shape(observables),
      biasInitializer)

    tf.stack(
      (0 until horizon).map(i => {
        input(---, i).tensorDot(weights, Seq(1), Seq(1)).add(bias)
      }),
      axis = -1)
  }
}
