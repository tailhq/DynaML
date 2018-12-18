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
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{ReuseOrCreateNew, ReuseExistingOnly}
import org.platanios.tensorflow.api.types.DataType

case class L2Regularization(
  names: Seq[String],
  dataTypes: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output, Output]("") {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {

    val weights = names.zip(dataTypes.zip(shapes)).map(n =>
      tf.variable(n._1, dataType = DataType.fromName(n._2._1), shape = n._2._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.square.sum()).reduce(_.add(_)).multiply(0.5*reg)

    input.add(reg_term)
  }
}

case class L1Regularization(
  names: Seq[String],
  dataTypes: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output, Output]("") {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {

    val weights = names.zip(dataTypes.zip(shapes)).map(n =>
      tf.variable(n._1, dataType = DataType.fromName(n._2._1), shape = n._2._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.abs.sum()).reduce(_.add(_)).multiply(reg)

    input.add(reg_term)
  }
}