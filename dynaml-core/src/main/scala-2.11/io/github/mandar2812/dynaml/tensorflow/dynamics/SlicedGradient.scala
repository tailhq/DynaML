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
package io.github.mandar2812.dynaml.tensorflow.dynamics

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api._

private[dynamics] case class SlicedGradient(
  override val name: String)(
  inputIndexer: Indexer*)(
  outputIndexer: Indexer*) extends
  TensorOperator[Output](name) {

  self =>

  private def gradTRec(y: Seq[Output], x: Output, rank: Int): Output = rank match {
    case 1 =>
      tf.stack(
        y.map(o => tf.gradients.gradients(Seq(o), Seq(x)).head.toOutput),
        axis = -1)
    case _ =>
      gradTRec(y.flatMap(_.unstack(-1, -1)), x, rank - 1)

  }

  override def sources: Map[String, Option[DifferentialOperator[Output, Output]]] = Map(self.name -> None)

  override def run(data: Layer[Output, Output]): Layer[Output, Output] = {

    new Layer[Output, Output](s"Grad[${data.name}]") {

      override val layerType: String = "GradLayer"

      override protected def _forward(input: Output)(implicit mode: Mode): Output = {

        val output = data.forward(input)

        val (final_input_shape, final_output_shape) = (
          Shape(input.shape.asArray.filterNot(_ == 1)),
          Shape(output.shape.asArray.filterNot(_ == 1))
        )

        val gradientShape =
          if(final_output_shape.rank > 1) final_input_shape ++ final_output_shape(1::)
          else final_input_shape
          /*if(output.shape(-1) == 1) input.shape ++ output.shape(1 :: -1)
          else input.shape ++ output.shape(1::)*/

        /*
        * Since there can be only one Ellipsis (---) when
        * slicing a tensor, check if both input and output
        * slices are set to ---, if so, return only a single
        * Ellipsis.
        * */
        val indexer =
          if(inputIndexer == Seq(---) && outputIndexer == Seq(---)) Seq(---)
          else Seq(::) ++ inputIndexer ++ outputIndexer

        val (unstacked_output, rank) =
          if(output.rank > 1) (tf.unstack(output, axis = -1), output.rank - 1)
          else (Seq(output), 1)

        val sliced_output = gradTRec(unstacked_output, input, rank).reshape(gradientShape)(indexer:_*)

        val finalShape = sliced_output.shape.asArray.filterNot(_ == 1)

        sliced_output.reshape(finalShape)
      }
    }
  }

}

/**
  * Implementation of the gradient operator (∇), see
  * [[_root_.io.github.mandar2812.dynaml.tensorflow.dynamics]],
  * for more details.
  * */
private[dynamics] object Gradient extends SlicedGradient(name = s"Grad")(---)(---)
