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

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * <h3>Compression &amp; Representation</h3>
  *
  * A layer which consists of compression and reconstruction modules.
  * Useful for unsupervised learning tasks.
  * */
case class AutoEncoder[I, J](
  override val name: String,
  encoder: Layer[I, J],
  decoder: Layer[J, I]) extends
  Layer[I, (J, I)](name) {

  self =>

  override val layerType: String = s"Representation[${encoder.layerType}, ${decoder.layerType}]"

  override def forwardWithoutContext(input: I)(implicit mode: Mode): (J, I) = {

    val encoding = encoder(input)

    val reconstruction = decoder(encoding)

    (encoding, reconstruction)
  }

  def >>[S](other: AutoEncoder[J, S]): ComposedAutoEncoder[I, J, S] = new ComposedAutoEncoder(name, self, other)
}

class ComposedAutoEncoder[I, J, K](
  override val name: String,
  encoder1: AutoEncoder[I, J],
  encoder2: AutoEncoder[J, K]) extends
  AutoEncoder[I, K](
    s"Compose[${encoder1.name}, ${encoder2.name}]",
    encoder1.encoder >> encoder2.encoder,
    encoder2.decoder >> encoder1.decoder)