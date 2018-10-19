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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.{Mode, TRAINING}
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables._

/**
  * Implementation of the <a href="https://arxiv.org/pdf/1502.03167.pdf">Batch Normalization</a> layer.
  * */
case class BatchNormalisation[T: TF: IsNotQuantized](override val name: String)
  extends Layer[Output[T], Output[T]](name) {

  override val layerType: String = s"BatchNorm"

  private val EPSILON = Tensor(1E-5f).toOutput.castTo[T]

  override def forwardWithoutContext(input: Output[T])(implicit mode: Mode): Output[T] = {
    val gamma      = tf.variable[T](
      "scaling", 
      input.shape(1::), OnesInitializer)

    val beta       = tf.variable[T](
      "offset",  
      input.shape(1::), ZerosInitializer)

    /*val running_mean      = tf.variable[T](
      "popmean", 
      input.shape(1::), ZerosInitializer)

    val running_var       = tf.variable[T](
      "popvar",  
      input.shape(1::), ZerosInitializer)

    val sample_count      = tf.variable[T](
      "samplecount",  INT32,
      Shape(), ZerosInitializer)*/

    val batch_mean = input.mean(axes = 0)
    val batch_var  = input.subtract(batch_mean).square.mean(axes = 0)

    //running_mean.assignAdd(batch_mean)
    //running_var.assignAdd(batch_var)
    //sample_count.assignAdd(1)

    input.subtract(batch_mean).divide(batch_var.add(EPSILON).sqrt).multiply(gamma).add(beta)
  }

  /*mode match {

    case TRAINING => {
      val gamma      = tf.variable[T](
        "scaling", 
        input.shape(1::), RandomUniformInitializer())

      val beta       = tf.variable[T](
        "offset",  
        input.shape(1::), RandomUniformInitializer())

      val running_mean      = tf.variable[T](
        "popmean", 
        input.shape(1::), ZerosInitializer)

      val running_var       = tf.variable[T](
        "popvar",  
        input.shape(1::), ZerosInitializer)

      val sample_count      = tf.variable[T](
        "samplecount",  INT32,
        Shape(), ZerosInitializer)

      val batch_mean = input.mean(axes = 0)
      val batch_var  = input.subtract(batch_mean).square.mean(axes = 0)

      running_mean.assignAdd(batch_mean)
      running_var.assignAdd(batch_var)
      sample_count.assignAdd(1)



      input.subtract(batch_mean).divide(batch_var.add(EPSILON).sqrt).multiply(gamma).add(beta)
    }

    case _ => {
      val gamma      = tf.variable[T](
        "scaling", 
        input.shape(1::), RandomUniformInitializer(),
        reuse = ReuseExistingOnly)

      val beta       = tf.variable[T](
        "offset",  
        input.shape(1::), RandomUniformInitializer(),
        reuse = ReuseExistingOnly)

      val running_mean      = tf.variable[T](
        "popmean", 
        input.shape(1::), ZerosInitializer,
        reuse = ReuseExistingOnly)

      val running_var       = tf.variable[T](
        "popvar",  
        input.shape(1::), ZerosInitializer,
        reuse = ReuseExistingOnly)

      val sample_count      = tf.variable[T](
        "samplecount",  INT32,
        Shape(), ZerosInitializer,
        reuse = ReuseExistingOnly)

      val (mean, variance): (Output, Output) =
        (running_mean.divide(sample_count), running_var.divide(sample_count))

      input.subtract(mean).divide(variance.add(EPSILON).sqrt).multiply(gamma).add(beta)
    }
  }*/
}
