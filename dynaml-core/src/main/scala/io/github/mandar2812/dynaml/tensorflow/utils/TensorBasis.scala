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

import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.spark.annotation.Experimental
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}

/**
  * A basis function expansion yielding a TF tensor.
  * */
@Experimental
case class TensorBasis[-I, D: TF: IsNotQuantized](f: I => Tensor[D]) extends DataPipe[I, Tensor[D]] {

  self =>

  override def run(data: I): Tensor[D] = f(data)

  def >[D1: TF: IsNotQuantized](other: DataPipe[Tensor[D], Tensor[D1]]): TensorBasis[I, D1] =
    TensorBasis((x: I) => other.run(self.f(x)))

}
