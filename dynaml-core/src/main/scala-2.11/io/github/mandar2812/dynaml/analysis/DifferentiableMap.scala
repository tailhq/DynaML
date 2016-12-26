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
package io.github.mandar2812.dynaml.analysis

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * A [[DataPipe]] which represents a differentiable transformation.
  * */
trait DifferentiableMap[S, D, J] extends DataPipe[S, D] {

  /**
    * Returns the Jacobian of the transformation
    * at the point x.
    * */
  def J(x: S): J
}

object DifferentiableMap {
  def apply[S, D, J](f: (S) => D, j: (S) => J): DifferentiableMap[S, D, J] =
    new DifferentiableMap[S, D, J] {
      /**
        * Returns the Jacobian of the transformation
        * at the point x.
        **/
      override def J(x: S) = j(x)

      override def run(data: S) = f(data)
    }
}