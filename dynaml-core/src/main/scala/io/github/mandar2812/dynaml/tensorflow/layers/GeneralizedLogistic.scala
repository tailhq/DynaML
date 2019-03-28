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
import org.platanios.tensorflow.api.core.types.{IsReal, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Activation
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, Regularizer}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * Generalized Logistic Function.
  * 
  * @author mandar2812 date 30/03/2018
  * */
case class GeneralizedLogistic[T: TF : IsReal](override val name: String) extends Activation(name) {

  override val layerType: String = "GeneralizedLogistic"

  override def forwardWithoutContext(input: Output[T])(implicit mode: Mode): Output[T] = {

    val alpha: tf.Variable[T] = tf.variable("alpha", Shape(input.shape(-1)), tf.RandomUniformInitializer())
    val nu:    tf.Variable[T] = tf.variable("nu",    Shape(input.shape(-1)), tf.OnesInitializer)
    val q:     tf.Variable[T] = tf.variable("Q",     Shape(input.shape(-1)), tf.OnesInitializer)
    val c:     tf.Variable[T] = tf.variable("C",     Shape(input.shape(-1)), tf.OnesInitializer)

    input
      .multiply(alpha.square.add(Tensor(1E-6).toOutput.castTo[T]).multiply(Tensor(-1.0).toOutput.castTo[T]))
      .exp
      .multiply(q.square)
      .add(c)
      .pow(nu.square.pow(Tensor(-1.0).toOutput.castTo[T]).multiply(Tensor(-1.0).toOutput.castTo[T]))
  }

}
