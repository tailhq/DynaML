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
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * Radial Basis Function Feedforward Layer.
  *
  * @param name Name of the layer.
  * @param num_units The number of centers/units.
  * @param centers_initializer The initialization of the node centers.
  * @param scales_initializer The initialization of the node length scales.
  * @param weights_initializer The initialization of the node importance weights.
  * @author mandar2812 date 30/03/2018
  * */
case class RBFLayer(
  override val name: String,
  num_units: Int,
  centers_initializer: Initializer = tf.RandomNormalInitializer(),
  scales_initializer: Initializer  = tf.OnesInitializer,
  weights_initializer: Initializer = tf.RandomNormalInitializer()) extends Layer[Output, Output](name) {

  override val layerType: String = s"RBFLayer[num_units:$num_units]"

  override def _forward(input: Output, mode: Mode): Output = {

    val node_centers    = tf.variable("node_centers", input.dataType, Shape(input.shape(-1), num_units), centers_initializer)

    val scales          = tf.variable("scales", input.dataType, Shape(input.shape(-1), num_units), scales_initializer)

    val weights         = tf.variable("weights", input.dataType, Shape(num_units), weights_initializer)

    val repeated_inputs = tf.stack(Seq.fill(num_units)(input), axis = -1)

    repeated_inputs
      .subtract(node_centers)
      .square
      .multiply(-1.0)
      .divide(scales.square.add(1E-6))
      .sum(axes = 1)
      .exp
      .multiply(weights)
  }
}
