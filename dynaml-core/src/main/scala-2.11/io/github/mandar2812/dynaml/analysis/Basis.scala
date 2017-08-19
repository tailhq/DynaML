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

import breeze.linalg._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils._

/**
  * Defines an abstract type for a basis expansion/mapping.
  * 
  * @author mandar2812 date 2017/08/18
  * */
abstract class Basis[I] extends DataPipe[I, DenseVector[Double]] {

  self =>

  protected val f: (I) => DenseVector[Double]

  override def run(x: I) = f(x)

  def >(other: DataPipe[DenseVector[Double], DenseVector[Double]]): Basis[I] = Basis((x: I) => other.run(self.run(x)))

  def *[J](other: Basis[J]): Basis[(I, J)] = Basis((x: (I, J)) => (self(x._1) * other(x._2).t).toDenseVector)

}

object Basis {

  /**
    * Create a basis, from a functional mapping.
    * */
  def apply[I](func: (I) => DenseVector[Double]): Basis[I] = new Basis[I] {

    val f = func
  }

}
