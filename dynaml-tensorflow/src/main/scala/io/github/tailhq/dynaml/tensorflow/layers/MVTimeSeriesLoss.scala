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
package io.github.tailhq.dynaml.tensorflow.layers

import org.platanios.tensorflow.api.core.types.{IsDecimal, IsFloatOrDouble, IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api._

/**
  * L2 loss for a time slice of a multivariate time series
  *
  * @author tailhq date 9/03/2018
  * */
case class MVTimeSeriesLoss[
Predictions: TF : IsNotQuantized : IsFloatOrDouble,
L: TF : IsFloatOrDouble](
  override val name: String)
  extends Loss[(Output[Predictions], Output[Predictions]), L](name) {
  override val layerType: String = "L2Loss"


  override def forwardWithoutContext(
    input: (Output[Predictions], Output[Predictions]))(
    implicit mode: Mode): Output[L] =
    tf.sum[Predictions, Int](
      tf.mean[Predictions, Int](
        tf.square(
          tf.subtract(input._1, input._2)),
        Tensor(0).toOutput)
    ).castTo[L]
}