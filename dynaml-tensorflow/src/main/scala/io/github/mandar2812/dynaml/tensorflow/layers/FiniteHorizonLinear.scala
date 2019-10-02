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

import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, Regularizer}
import org.platanios.tensorflow.api.{---, Output, Shape, tf}

/**
  * Projection of a finite horizon multivariate
  * time series onto an observation space.
  *
  * @param observables The dimensionality of the observations at each time epoch.
  * @author mandar2812 date 11/03/2018
  * */
case class FiniteHorizonLinear[T: TF: IsNotQuantized](
  override val name: String, observables: Int,
  weightsInitializer: Initializer = RandomNormalInitializer(),
  biasInitializer: Initializer = RandomNormalInitializer(),
  regularization: Regularizer = null) extends
  Layer[Output[T], Output[T]](name) {

  override val layerType: String = s"FHLinear[observables:$observables]"

  override def forwardWithoutContext(input: Output[T])(implicit mode: Mode): Output[T] = {

    val units        = input.shape(-2)

    val horizon      = input.shape(-1)

    val weights      = tf.variable[T](
      "Weights", Shape(observables, units),
      weightsInitializer, regularizer = regularization)

    val bias         = tf.variable[T](
      "Bias", Shape(observables),
      biasInitializer)

    tf.stack(
      (0 until horizon).map(i => {
        input(---, i).tensorDot(weights, Seq(1), Seq(1)).add(bias)
      }),
      axis = -1)
  }
}
