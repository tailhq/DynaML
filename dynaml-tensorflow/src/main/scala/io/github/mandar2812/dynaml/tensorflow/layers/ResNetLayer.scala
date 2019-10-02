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
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * <h3>Compression &amp; Representation</h3>
  *
  * A layer which maps x -> (f(x), x)
  * */
case class ResNetLayer[I, J](
  override val name: String,
  f: Layer[I, J]) extends
  Layer[I, (J, I)](name) {

  self =>

  override val layerType: String = s"ResNet[${f.layerType}]"

  override def forwardWithoutContext(input: I)(implicit mode: Mode): (J, I) = {

    val f_x = f.forwardWithoutContext(input)

    (f_x, input)
  }

  def >>[S](other: ResNetLayer[J, S]): ResNetLayer[I, S] = ResNetLayer(s"${self.name}[${other.name}]", self.f >> other.f)
}