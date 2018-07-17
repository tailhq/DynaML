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
package io.github.mandar2812.dynaml.pipes
import scalaxy.streams.optimize


/**
  * @author mandar2812 on 17/11/15.
  *
  * Represents an abstract stream data pipeline.
  * @tparam I The type of a single source record
  * @tparam J The result type of a single record.
  *
  */
trait IterableDataPipe[I, J, K] extends DataPipe[Iterable[I], K]{

  /**
    * The functions that compose the
    * pipe operations.
    *
    * */
  val pipe: (I) => J

  /**
    * The function which writes
    * to the destination
    * */

  override def run(data: Iterable[I]): K
}

/**
  * A pipeline which takes a [[Iterable]] of data and
  * performs the scala `map`operation.
  * */
trait IterableMapPipe[I, J] extends IterableDataPipe[I, J, Iterable[J]] {
  override def run(data: Iterable[I]): Iterable[J] = optimize { data.map(pipe) }
}

/**
  * A pipeline which takes a [[Iterable]] of data and
  * performs the scala `flatMap` operation.
  * */
trait IterableFlatMapPipe[I, J] extends IterableDataPipe[I, Iterable[J], Iterable[J]] {
  override def run(data: Iterable[I]) = optimize { data.flatMap(pipe) }
}

trait IterableFilterPipe[I] extends IterableDataPipe[I, Boolean, Iterable[I]] {
  override def run(data: Iterable[I]): Iterable[I] = optimize { data.filter(pipe) }
}

trait IterablePartitionPipe[I] extends IterableDataPipe[I, Boolean, (Iterable[I], Iterable[I])] {
  override def run(data: Iterable[I]): (Iterable[I], Iterable[I]) = optimize { data.partition(pipe) }
}

trait IterableSideEffectPipe[I] extends IterableDataPipe[I, Unit, Unit] {
  override def run(data: Iterable[I]): Unit = data.foreach(pipe)
}

object IterableDataPipe {

  def toIterablePipe[I, S <: Traversable[I]] =
    new DataPipe[S, Iterable[I]] {
      override def run(data: S) = data.toIterable
    }

  //Iterable pipes which map from the original domain to a new one
  def apply[I, J](mapFunc: (I) => J): IterableMapPipe[I, J] =
    new IterableMapPipe[I, J] {
      val pipe = mapFunc
    }

  def apply[I, J](map: DataPipe[I, J]): IterableMapPipe[I, J] =
    new IterableMapPipe[I, J] {
      val pipe = map.run _
    }

  //Iterable pipes which act as filters
  def apply[I](mapFunc: (I) => Boolean): IterableFilterPipe[I] =
    new IterableFilterPipe[I] {
      val pipe = mapFunc
    }

  def apply[I](mapFunc: DataPipe[I, Boolean]): IterableFilterPipe[I] =
    new IterableFilterPipe[I] {
      val pipe = mapFunc.run _
    }

  //stream pipes with side effects
  def apply[I](seFunc: (I) => Unit): IterableSideEffectPipe[I] =
    new IterableSideEffectPipe[I] {
      val pipe = seFunc
    }

  def apply[I](seFunc: SideEffectPipe[I]): IterableSideEffectPipe[I] =
    new IterableSideEffectPipe[I] {
      val pipe = seFunc.run _
    }
}

object IterableFlatMapPipe {
  def apply[I, J](mapFunc: (I) => Iterable[J]) =
    new IterableFlatMapPipe[I, J] {
      override val pipe = mapFunc
    }

  def apply[I, J](mapFunc: DataPipe[I, Iterable[J]]) =
    new IterableFlatMapPipe[I, J] {
      override val pipe = mapFunc.run _
    }

}

object IterablePartitionPipe {
  def apply[I](mapFunc: (I) => Boolean): IterablePartitionPipe[I] =
    new IterablePartitionPipe[I] {
      val pipe = mapFunc
    }

  def apply[I](mapFunc: DataPipe[I, Boolean]): IterablePartitionPipe[I] =
    new IterablePartitionPipe[I] {
      val pipe = mapFunc.run _
    }
}