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
package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps}

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date: 25/10/2016.
  */
abstract class AbstractPartitionedVector[T](
  data: Stream[(Long, T)],
  num_row_blocks: Long = -1L)
  extends NumericOps[AbstractPartitionedVector[T]] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.map(_._1).max + 1L else num_row_blocks

  lazy val colBlocks = 1L

  lazy val cols: Long = 1L

  def _data = data.sortBy(_._1)

  override def repr: AbstractPartitionedVector[T] = this

  def map(func: ((Long, T)) => (Long, T)): AbstractPartitionedVector[T]

  def apply(r: NumericRange[Long]): AbstractPartitionedVector[T]


}

object AbstractPartitionedVector {

  /**
    * Create a [[PartitionedVector]] from a breeze [[DenseVector]]
    * @param v input vector
    * @param num_elements_per_block The size of each block
    * @return A [[PartitionedVector]] instance.
    */
  def apply(v: DenseVector[Double], num_elements_per_block: Int): PartitionedVector = {
    val blocks = v.toArray
      .grouped(num_elements_per_block)
      .zipWithIndex
      .map(c => (c._2.toLong, DenseVector(c._1)))
      .toStream

    new PartitionedVector(blocks)
  }

}