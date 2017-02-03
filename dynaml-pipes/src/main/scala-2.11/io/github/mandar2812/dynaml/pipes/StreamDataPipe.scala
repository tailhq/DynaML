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
trait StreamDataPipe[I, J, K] extends DataPipe[Stream[I], K]{

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

  override def run(data: Stream[I]): K
}

/**
  * A pipeline which takes a [[Stream]] of data and
  * performs the scala `map`operation.
  * */
trait StreamMapPipe[I, J] extends StreamDataPipe[I, J, Stream[J]] {
  override def run(data: Stream[I]): Stream[J] = optimize { data.map(pipe) }
}

/**
  * A pipeline which takes a [[Stream]] of data and
  * performs the scala `flatMap` operation.
  * */
trait StreamFlatMapPipe[I, J] extends StreamDataPipe[I, Stream[J], Stream[J]] {
  override def run(data: Stream[I]) = optimize { data.flatMap(pipe) }
}

trait StreamFilterPipe[I] extends StreamDataPipe[I, Boolean, Stream[I]] {
  override def run(data: Stream[I]): Stream[I] = optimize { data.filter(pipe) }
}

trait StreamPartitionPipe[I] extends StreamDataPipe[I, Boolean, (Stream[I], Stream[I])] {
  override def run(data: Stream[I]): (Stream[I], Stream[I]) = optimize { data.partition(pipe) }
}

trait StreamSideEffectPipe[I] extends StreamDataPipe[I, Unit, Unit] {
  override def run(data: Stream[I]): Unit = data.foreach(pipe)
}

object StreamDataPipe {

  //Stream pipes which map from the original domain to a new one
  def apply[I, J](mapFunc: (I) => J): StreamMapPipe[I, J] =
    new StreamMapPipe[I, J] {
      val pipe = mapFunc
    }

  def apply[I, J](map: DataPipe[I, J]): StreamMapPipe[I, J] =
    new StreamMapPipe[I, J] {
      val pipe = map.run _
    }

  //Stream pipes which act as filters
  def apply[I](mapFunc: (I) => Boolean): StreamFilterPipe[I] =
    new StreamFilterPipe[I] {
      val pipe = mapFunc
    }

  def apply[I](mapFunc: DataPipe[I, Boolean]): StreamFilterPipe[I] =
    new StreamFilterPipe[I] {
      val pipe = mapFunc.run _
    }

  //stream pipes with side effects
  def apply[I](seFunc: (I) => Unit): StreamSideEffectPipe[I] =
    new StreamSideEffectPipe[I] {
      val pipe = seFunc
    }

  def apply[I](seFunc: SideEffectPipe[I]): StreamSideEffectPipe[I] =
    new StreamSideEffectPipe[I] {
      val pipe = seFunc.run _
    }
}

object StreamFlatMapPipe {
  def apply[I, J](mapFunc: (I) => Stream[J]) =
    new StreamFlatMapPipe[I, J] {
      override val pipe = mapFunc
    }

  def apply[I, J](mapFunc: DataPipe[I, Stream[J]]) =
    new StreamFlatMapPipe[I, J] {
      override val pipe = mapFunc
    }

}

object StreamPartitionPipe {
  def apply[I](mapFunc: (I) => Boolean): StreamPartitionPipe[I] =
    new StreamPartitionPipe[I] {
      val pipe = mapFunc
    }

  def apply[I](mapFunc: DataPipe[I, Boolean]): StreamPartitionPipe[I] =
    new StreamPartitionPipe[I] {
      val pipe = mapFunc.run _
    }
}