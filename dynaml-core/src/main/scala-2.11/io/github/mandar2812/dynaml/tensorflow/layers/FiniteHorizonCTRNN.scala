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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, Regularizer}

/**
  * Represents a Continuous Time Recurrent Neural Network (CTRNN)
  * The layer simulates the discretized dynamics of the CTRNN for
  * a fixed number of time steps.
  *
  * @author mandar2812 date: 2018/03/06
  * */
case class FiniteHorizonCTRNN(
  override val name: String, units: Int,
  horizon: Int, timestep: Double,
  weightsInitializer: Initializer = RandomNormalInitializer(),
  biasInitializer: Initializer = RandomNormalInitializer(),
  gainInitializer: Initializer = RandomNormalInitializer(),
  timeConstantInitializer: Initializer = RandomNormalInitializer(),
  regularization: Regularizer = L2Regularizer()) extends
  Layer[Output, Output](name) {

  override val layerType: String = "FHCTRNN"

  override protected def _forward(input: Output, mode: Mode): Output = {

    val weights      = tf.variable(
      "Weights", input.dataType, Shape(units, units),
      weightsInitializer, regularizer = regularization)

    val timeconstant = tf.variable(
      "TimeConstant", input.dataType, Shape(units, units),
      timeConstantInitializer)

    val gain         = tf.variable(
      "Gain", input.dataType, Shape(units, units),
      timeConstantInitializer, regularizer = regularization)

    val bias         = tf.variable(
      "Bias", input.dataType, Shape(units),
      biasInitializer)

    tf.stack(
      (1 to horizon).scanLeft(input)((x, _) => {
        val decay = x.tensorDot(timeconstant.multiply(-1d), Seq(1), Seq(0))
        val interaction = x.tensorDot(gain, Seq(1), Seq(0)).add(bias).tanh.tensorDot(weights, Seq(1), Seq(0))

        x.add(decay.multiply(timestep)).add(interaction.multiply(timestep))
      }).tail,
      axis = -1)

  }
}
