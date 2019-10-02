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

import org.platanios.tensorflow.api.core.types.{IsReal, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Activation
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

/**
  * The cumulative gaussian function, as a
  * Tensorflow activation.
  * */
case class Phi[T: TF : IsReal](override val name: String)
  extends Activation[T](name) {
  override val layerType: String = "Phi"

  override def forwardWithoutContext(input: Output[T])(implicit mode: Mode): Output[T] =
    ops.Math.erf(input.divide(Tensor(math.sqrt(2.0f)).toOutput.castTo[T]))
      .add(Tensor(1.0f).toOutput.castTo[T])
      .multiply(Tensor(0.5f).toOutput.castTo[T])
      .castTo[T]
}