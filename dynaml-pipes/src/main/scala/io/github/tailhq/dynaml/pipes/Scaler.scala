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
package io.github.tailhq.dynaml.pipes
import com.fasterxml.jackson.module.scala.deser.`package`.overrides

/**
  * @author tailhq 17/6/16.
  *
  * Top level trait; represents the scaling operation, used
  * heavily in data processing tasks.
  */
trait Scaler[S] extends DataPipe[S, S] {
  override def apply[T <: Traversable[S]](data: T) =
    data.map(run).asInstanceOf[T]

  def *[T](that: Scaler[T]) = {
    val firstRun = this.run _
    new Scaler[(S, T)] {
      override def run(data: (S, T)): (S, T) =
        (firstRun(data._1), that(data._2))
    }
  }

  def >(otherScaler: Scaler[S]) = {

    val firstRun = this.run _

    new Scaler[S] {
      def run(data: S) = otherScaler.run(firstRun(data))
    }
  }

}

object Scaler {
  def apply[S](f: S => S): Scaler[S] =
    new Scaler[S] {
      override def run(data: S): S = f(data)
    }

  def apply[S](f: DataPipe[S, S]): Scaler[S] = apply(f.run _)

  def apply[I](f: DataPipe[I, I], r: DataPipe[I, I]): ReversibleScaler[I] =
    new ReversibleScaler[I] {

      override val i: Scaler[I] = Scaler(r)

      override def run(data: I): I = f(data)

    }
}

/**
  * Performs a reversible scaling operation.
  *
  * @author tailhq 17/6/16
  * */
trait ReversibleScaler[S] extends Scaler[S] with Encoder[S, S] {

  self =>

  /**
    * The inverse operation of this scaling.
    *
    * */
  override val i: Scaler[S]

  override def apply[T <: Traversable[S]](data: T): T =
    data.map(run).asInstanceOf[T]

  def *[T](other: ReversibleScaler[T]) = ReversibleScalerTuple(
    self,
    other
  )

  def >(other: ReversibleScaler[S]): ReversibleScaler[S] =
    ComposedReversibleScaler(self, other)
}

case class ReversibleScalerTuple[I, J](
  val _1: ReversibleScaler[I],
  val _2: ReversibleScaler[J])
    extends ReversibleScaler[(I, J)] {

  override val i: Scaler[(I, J)] = _1.i * _2.i

  override def run(data: (I, J)): (I, J) = (_1(data._1), _2(data._2))
}

case class ComposedReversibleScaler[I](
  _1: ReversibleScaler[I],
  _2: ReversibleScaler[I])
    extends ReversibleScaler[I] {

  override val i: Scaler[I] = _2.i > _1.i

  override def run(data: I): I = _2(_1(data))
}
