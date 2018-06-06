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

import org.platanios.tensorflow.api.{Shape, tf}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer

/**
  * Implementation of the <a href="https://arxiv.org/pdf/1502.03167.pdf">Batch Normalization</a> layer.
  *
  * @param axis The axis over which the features should be normalised, defaults to 0
  * */
case class BatchNormalisation(override val name: String, axis: Int = 0)
  extends Layer[Output, Output](name) {

  override val layerType: String = s"BatchNorm"

  private val EPSILON = 1E-5

  override protected def _forward(input: Output, mode: Mode): Output = {

    val gamma      = tf.variable("scaling", input.dataType, input.shape, ConstantInitializer(1.0))
    val beta       = tf.variable("offset",  input.dataType, input.shape, ConstantInitializer(0.0))

    val batch_mean = input.mean(axes = axis)
    val batch_var  = input.subtract(batch_mean).square.mean(axes = axis)

    input.subtract(batch_mean).divide(batch_var.add(EPSILON).sqrt).multiply(gamma).add(beta)
  }
}
