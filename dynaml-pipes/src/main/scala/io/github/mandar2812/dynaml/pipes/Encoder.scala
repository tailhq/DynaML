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
  * A deterministic and reversible encoding
  * from a domain to a range. Mathematically equivalent
  * to a bijective function.
  * @tparam S The domain type
  * @tparam D The output type
  *
  * @author mandar2812 date 23/10/2016.
  *
  * */
trait Encoder[S, D] extends DataPipe[S, D] {

  /**
    * Represents the decoding operation.
    * */
  val i: DataPipe[D, S]

  /**
    * Represents the composition of two
    * encoders, resulting in a third encoder
    * Schematically represented as:
    *
    * [[S]] -> [[D]] :: [[D]] -> [[Further]] ==
    * [[S]] -> [[Further]]
    *
    * */
  def >[Further](that: Encoder[D, Further]): Encoder[S, Further] = {
    val fPipe1 = DataPipe(this.run _)

    val rPipe1 = this.i

    val fPipe2 = DataPipe(that.run _)

    val rPipe2 = that.i

    val fPipe = fPipe1 > fPipe2
    val rPipe = rPipe2 > rPipe1

    Encoder(fPipe, rPipe)
  }
}

object Encoder {

  /**
    * Create an encoder on the fly by supplying the encode and decode function
    * */
  def apply[S, D](forwardEnc: S => D, reverseEnc: D => S): Encoder[S, D] =
    new Encoder[S, D] {

      val i = DataPipe(reverseEnc)

      override def run(data: S): D = forwardEnc(data)

    }

  def apply[S, D](forwardPipe: DataPipe[S, D], reversePipe: DataPipe[D, S]) =
    new Encoder[S, D] {

      val i = reversePipe

      override def run(data: S) = forwardPipe(data)

    }
}