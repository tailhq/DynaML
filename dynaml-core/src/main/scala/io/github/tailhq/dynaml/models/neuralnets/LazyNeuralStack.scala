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
package io.github.tailhq.dynaml.models.neuralnets

import scala.collection.GenTraversableLike


/**
  * @author tailhq date: 17/04/2017.
  *
  * A computation stack whose layers are lazily computed
  * */
class LazyNeuralStack[P, I](elements: Stream[NeuralLayer[P, I, I]]) extends
  GenericNeuralStack[P, I, NeuralLayer[P, I, I], Stream](elements) {

  self =>

  /**
    * Slice the stack according to a range.
    **/
  override def apply(r: Range): LazyNeuralStack[P, I] = new LazyNeuralStack[P, I](g.slice(r.min, r.max + 1))


  /**
    * Append another computation stack to the end of the
    * current one.
    **/
  override def ++[
  L <: NeuralLayer[P, I, I],
  G[L1] <: Traversable[L1] with GenTraversableLike[L1, G[L1]]](
    otherStack: GenericNeuralStack[P, I, L, G]) =
    new LazyNeuralStack[P, I](self.g ++ otherStack._layers.asInstanceOf[Stream[NeuralLayer[P, I, I]]])

  /**
    * Append a single computation layer to the stack.
    **/
  override def :+(computationLayer: NeuralLayer[P, I, I]): LazyNeuralStack[P, I] =
    new LazyNeuralStack[P, I](self.g :+ computationLayer)
}

object LazyNeuralStack {

  def apply[P, I](
    layerFunc: (Int) => NeuralLayer[P, I, I],
    num_layers: Int) = new LazyNeuralStack[P, I](
    (0 until num_layers).toStream.map(i => layerFunc(i))
  )
}