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
package io.github.tailhq.dynaml.tensorflow.utils

import org.platanios.tensorflow.api._
import _root_.io.github.tailhq.dynaml.pipes._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}

/**
  * Scales attributes of a vector pattern using the sample mean and variance of
  * each dimension. This assumes that there is no covariance between the data
  * dimensions.
  *
  * @param mean Sample mean of the data
  * @param sigma Sample variance of each data dimension
  * @author tailhq date: 07/03/2018.
  *
  * */
case class GaussianScalerTF[D : TF: IsNotQuantized](mean: Tensor[D], sigma: Tensor[D]) extends TFScaler[D] {

  override val i: Scaler[Tensor[D]] = Scaler((xc: Tensor[D]) => tfi.add(tfi.multiply(xc, sigma), mean))

  override def run(data: Tensor[D]): Tensor[D] = tfi.divide(tfi.subtract(data, mean), sigma)

  def apply(indexers: Indexer*): GaussianScalerTF[D] = this.copy(
    mean(indexers.head, indexers.tail:_*),
    sigma(indexers.head, indexers.tail:_*))

}


case class GaussianScalerTO[D : TF: IsNotQuantized](mean: Output[D], sigma: Output[D]) extends TOScaler {

  override val i: Scaler[Output[D]] = Scaler((xc: Output[D]) => xc.multiply(sigma).add(mean))

  override def run(data: Output[D]): Output[D] = data.subtract(mean).divide(sigma)

  def apply(indexers: Indexer*): GaussianScalerTO[D] = this.copy(
    mean(indexers.head, indexers.tail:_*),
    sigma(indexers.head, indexers.tail:_*))

}
