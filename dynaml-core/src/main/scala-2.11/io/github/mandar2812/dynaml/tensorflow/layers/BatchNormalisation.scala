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

import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, RandomUniformInitializer, ReuseExistingOnly, ZerosInitializer}

/**
  * Implementation of the <a href="https://arxiv.org/pdf/1502.03167.pdf">Batch Normalization</a> layer.
  * */
case class BatchNormalisation(override val name: String)
  extends Layer[Output, Output](name) {

  override val layerType: String = s"BatchNorm"

  private val EPSILON = 1E-5

  override protected def _forward(input: Output, mode: Mode): Output = {

    val gamma      = tf.variable(
      "scaling", input.dataType,
      input.shape(1::), RandomUniformInitializer())

    val beta       = tf.variable(
      "offset",  input.dataType,
      input.shape(1::), RandomUniformInitializer())

    val (mean, variance): (Output, Output) = if (mode.isTraining) {

      val running_mean      = tf.variable(
        "popmean", input.dataType,
        input.shape(1::), ZerosInitializer)

      val running_var       = tf.variable(
        "popvar",  input.dataType,
        input.shape(1::), ZerosInitializer)

      val sample_count      = tf.variable(
        "samplecount",  INT32,
        Shape(), ZerosInitializer)

      val batch_mean = input.mean(axes = 0)
      val batch_var  = input.subtract(batch_mean).square.mean(axes = 0)

      running_mean.assignAdd(batch_mean)
      running_var.assignAdd(batch_var)
      sample_count.assignAdd(1)

      (batch_mean, batch_var)
    } else {
      val running_mean      = tf.variable(
        "popmean", input.dataType,
        input.shape(1::), ZerosInitializer,
        reuse = ReuseExistingOnly)

      val running_var       = tf.variable(
        "popvar",  input.dataType,
        input.shape(1::), ZerosInitializer,
        reuse = ReuseExistingOnly)

      val sample_count      = tf.variable(
        "samplecount",  INT32,
        Shape(), ZerosInitializer,
        reuse = ReuseExistingOnly)

      (running_mean.divide(sample_count), running_var.divide(sample_count))
    }



    input.subtract(mean).divide(variance.add(EPSILON).sqrt).multiply(gamma).add(beta)
  }
}
