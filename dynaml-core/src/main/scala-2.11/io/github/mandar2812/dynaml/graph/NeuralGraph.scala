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
package io.github.mandar2812.dynaml.graph

/**
  * A Neural Graph is an encapsulation
  * of a base graph of type [[B]] with
  * a set of predefined behaviors.
  *
  * @tparam B The type of the encapsulated
  *           base graph object
  *
  * @tparam I Type of input
  * @tparam J Type of output
  * */
trait NeuralGraph[B, I, J] {
  protected val g: B
  val forwardPass: (I) => J
}
