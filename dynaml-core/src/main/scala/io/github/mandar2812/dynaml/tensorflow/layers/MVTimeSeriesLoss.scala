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
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops.Output

/**
  * L2 loss for a time slice of a multivariate time series
  *
  * @author mandar2812 date 9/03/2018
  * */
case class MVTimeSeriesLoss(override val name: String)
  extends Loss[(Output, Output)](name) {
  override val layerType: String = "L2Loss"

  override protected def _forward(input: (Output, Output))(implicit mode: Mode): Output = {
    input._1.subtract(input._2).square.mean(axes = 0).sum()
  }
}