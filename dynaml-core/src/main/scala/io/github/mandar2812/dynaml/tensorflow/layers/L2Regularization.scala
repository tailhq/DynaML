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
package io.github.mandar2812.dynaml.tensorflow.layers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, IsReal, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.ReuseExistingOnly



case class L2Regularization[D: TF: IsNotQuantized](
  names: Seq[String],
  dataTypes: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output[D], Output[D]]("") {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  @throws[IllegalArgumentException]
  @throws[ShapeMismatchException]
  @throws[InvalidDataTypeException]
  override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Output[D] = {

    val weights = names.zip(dataTypes.zip(shapes)).map(n =>
      tf.variable[D](n._1, shape = n._2._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.square.sum[Int]()).reduce(_.add(_)).multiply(Tensor(0.5*reg).toOutput.castTo[D])

    input.add(reg_term)
  }
}


case class L1Regularization[D: TF: IsNotQuantized: IsReal](
  names: Seq[String],
  dataTypes: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output[D], Output[D]]("") {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  @throws[IllegalArgumentException]
  @throws[ShapeMismatchException]
  @throws[InvalidDataTypeException]
  override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Output[D] = {

    val weights = names.zip(dataTypes.zip(shapes)).map(n =>
      tf.variable(n._1, shape = n._2._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.abs.sum[Int]()).reduce(_.add(_)).multiply(Tensor(reg).toOutput.castTo[D])

    input.add(reg_term)
  }
}