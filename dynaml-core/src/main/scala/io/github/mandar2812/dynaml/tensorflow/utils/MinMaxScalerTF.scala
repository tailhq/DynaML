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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}


/**
  * Scales attributes of a vector pattern using the sample minimum and maximum of
  * each dimension.
  *
  * @param min Sample minimum of the data
  * @param max Sample maximum of each data dimension
  * @author mandar2812 date: 07/03/2018.
  *
  * */
case class MinMaxScalerTF[D: TF: IsNotQuantized](min: Tensor[D], max: Tensor[D]) extends TFScaler[D] {

  val delta: Tensor[D] = tfi.subtract(max, min)

  override val i: Scaler[Tensor[D]] = Scaler((xc: Tensor[D]) => tfi.add(tfi.multiply(xc, delta), min))

  override def run(data: Tensor[D]): Tensor[D] = data.subtract(min).divide(delta)

  def apply(indexers: Indexer*): MinMaxScalerTF[D] = this.copy(
    min(indexers.head, indexers.tail:_*),
    max(indexers.head, indexers.tail:_*))


}

case class MinMaxScalerTO[D: TF: IsNotQuantized](min: Output[D], max: Output[D]) extends TOScaler {

  val delta: Output[D] = max.subtract(min)

  override val i: Scaler[Output[D]] = Scaler((xc: Output[D]) => xc.multiply(delta).add(min))

  override def run(data: Output[D]): Output[D] = data.subtract(min).divide(delta)

  def apply(indexers: Indexer*): MinMaxScalerTO[D] = this.copy(
    min(indexers.head, indexers.tail:_*),
    max(indexers.head, indexers.tail:_*))

}
