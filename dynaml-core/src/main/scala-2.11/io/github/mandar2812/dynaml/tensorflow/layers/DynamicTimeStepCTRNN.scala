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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables._


/**
  * <h3>Continuous Time Recurrent Neural Network</h3>
  * <br/>
  * <h4>With with time step inference.</h4>
  *
  * Represents a Continuous Time Recurrent Neural Network (CTRNN)
  * The layer simulates the discretized dynamics of the CTRNN for
  * a fixed number of time steps.
  *
  * A variant of [[FiniteHorizonCTRNN]], here the integration time-step
  * is inferred during training.
  *
  * @author mandar2812 date: 2018/03/06
  * */
case class DynamicTimeStepCTRNN[T: TF: IsNotQuantized](
  override val name: String, horizon: Int,
  weightsInitializer: Initializer = RandomNormalInitializer(),
  biasInitializer: Initializer = RandomNormalInitializer(),
  gainInitializer: Initializer = RandomNormalInitializer(),
  timeConstantInitializer: Initializer = RandomNormalInitializer(),
  regularization: Regularizer = null) extends
  Layer[Output[T], Output[T]](name) {

  override val layerType: String = s"CTRNN[horizon:$horizon]"

  override def forwardWithoutContext(input: Output[T])(implicit mode: Mode): Output[T] = {

    val timestep     = tf.variable[T]("time_step", Shape(), new RandomUniformInitializer)

    val units        = input.shape(-1)


    val weights      = tf.variable[T](
      "Weights", Shape(units, units),
      weightsInitializer, regularizer = regularization)

    val timeconstant = tf.variable[T](
      "TimeConstant", Shape(units, units),
      timeConstantInitializer, regularizer = regularization)

    val gain         = tf.variable[T](
      "Gain", Shape(units, units),
      timeConstantInitializer, regularizer = regularization)

    val bias         = tf.variable[T](
      "Bias", Shape(units),
      biasInitializer, regularizer = regularization)

    tf.stack(
      (1 to horizon).scanLeft(input)((x, _) => {
        val decay = x.tensorDot(timeconstant.multiply(Tensor(-1).toOutput.castTo[T]), Seq(1), Seq(0))
        val interaction = x.tensorDot(gain, Seq(1), Seq(0)).add(bias).tanh.tensorDot(weights, Seq(1), Seq(0))

        x.add(decay.multiply(timestep)).add(interaction.multiply(timestep))
      }).tail,
      axis = -1)

  }
}
