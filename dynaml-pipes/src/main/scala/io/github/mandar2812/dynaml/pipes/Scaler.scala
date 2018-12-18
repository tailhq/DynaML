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

/**
  * @author mandar2812 17/6/16.
  *
  * Top level trait; represents the scaling operation, used
  * heavily in data processing tasks.
  */
trait Scaler[S] extends DataPipe[S, S]{
  override def apply[T <: Traversable[S]](data: T) =
    data.map(run).asInstanceOf[T]

  def *[T](that: Scaler[T]) = {
    val firstRun = this.run _
    new Scaler[(S,T)] {
      override def run(data: (S, T)): (S, T) = (firstRun(data._1), that(data._2))
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
  def apply[S](f: (S) => S): Scaler[S] =
    new Scaler[S] {
      override def run(data: S): S = f(data)
    }
}

/**
  * @author mandar2812 17/6/16
  *
  *
  * */
trait ReversibleScaler[S] extends Scaler[S] with Encoder[S, S]{

  /**
    * The inverse operation of this scaling.
    *
    * */
  override val i: Scaler[S]

  override def apply[T<: Traversable[S]](data: T):T =
    data.map(run).asInstanceOf[T]

  def *[T](that: ReversibleScaler[T]) = {

    val firstInv = this.i

    val firstRun = this.run _

    new ReversibleScaler[(S, T)] {

      val i: Scaler[(S,T)] = firstInv * that.i

      override def run(data: (S, T)): (S, T) = (firstRun(data._1), that(data._2))
    }
  }

  def >(otherRevScaler: ReversibleScaler[S]): ReversibleScaler[S] = {

    val firstInv = this.i

    val firstRun = this.run _

    new ReversibleScaler[S] {
      val i: Scaler[S] = otherRevScaler.i > firstInv
      def run(data: S) = otherRevScaler.run(firstRun(data))
    }
  }
}