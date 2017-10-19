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

import breeze.linalg.NumericOps

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date: 25/10/2016.
  */
abstract class AbstractPartitionedMatrix[T](
  data: Stream[((Long, Long), T)],
  num_row_blocks: Long = -1L,
  num_col_blocks: Long = -1L)
  extends NumericOps[AbstractPartitionedMatrix[T]] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.map(_._1._1).max + 1L else num_row_blocks

  lazy val colBlocks = if(num_col_blocks == -1L) data.map(_._1._2).max + 1L else num_col_blocks

  def _data: Stream[((Long, Long), T)] = data.sortBy(_._1)

  override def repr: AbstractPartitionedMatrix[T] = this

  def t: AbstractPartitionedMatrix[T]

  def map(f: (((Long, Long), T)) => ((Long, Long), T)): AbstractPartitionedMatrix[T]

  /**
    * Slice a blocked matrix to produce a new block matrix.
    */
  def apply(r: NumericRange[Long], c: NumericRange[Long]): AbstractPartitionedMatrix[T]

  def filterBlocks(f: ((Long, Long)) => Boolean): Stream[((Long, Long), T)] =
    _data.filter(c => f(c._1))


}
