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
package io.github.mandar2812.dynaml.tensorflow.utils

import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.pipes._

/**
  * Scales attributes of a vector pattern using the sample minimum and maximum of
  * each dimension.
  *
  * @param min Sample minimum of the data
  * @param max Sample maximum of each data dimension
  * @author mandar2812 date: 07/03/2018.
  *
  * */
case class MinMaxScalerTF(min: Tensor, max: Tensor) extends TFScaler {

  val delta: Tensor = max.subtract(min)

  override val i: Scaler[Tensor] = Scaler((xc: Tensor) => xc.multiply(delta).add(min))

  override def run(data: Tensor): Tensor = data.subtract(min).divide(delta)

}
