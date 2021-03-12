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
package io.github.tailhq.dynaml.tensorflow.dynamics

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}

private[dynamics] case class SlicedGradient[D: TF: IsNotQuantized](
  override val name: String)(
  val inputIndexer: Indexer*)(
  val outputIndexer: Indexer*) extends
  TensorOperator[Output[D], D](name) {

  self =>

  private def gradTRec(y: Seq[Output[D]], x: Output[D], rank: Int): Output[D] = rank match {
    case 1 =>
      tf.stack(
        y.map(o => tf.gradients.gradients[D, D](Seq(o), Seq(x), TF[D].dataType).head.toOutput),
        axis = -1)
    case _ =>
      gradTRec(y.flatMap(_.unstack(-1, -1)), x, rank - 1)

  }

  protected[dynamics] override def sources: Map[String, Option[DifferentialOperator[Output[D], Output[D]]]] =
    Map(self.name -> None)

  override def run(data: Layer[Output[D], Output[D]]): Layer[Output[D], Output[D]] = {

    new Layer[Output[D], Output[D]](s"Grad_${data.name}_") {

      override val layerType: String = "GradLayer"

      override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Output[D] = {

        val output = data.forward(input)

        val (final_input_shape, final_output_shape) = (
          Shape(input.shape.asArray.filterNot(_ == 1)),
          Shape(output.shape.asArray.filterNot(_ == 1))
        )

        val gradientShape =
          if(final_output_shape.rank > 1) final_input_shape ++ final_output_shape(1::)
          else final_input_shape

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

        val sliced_output = gradTRec(unstacked_output, input, rank)
          .reshape(gradientShape.toOutput)
          .slice(indexer.head, indexer.tail:_*)

        val finalShape = sliced_output.shape.asArray.filterNot(_ == 1)

        sliced_output.reshape(finalShape)
      }
    }
  }

}

/**
  * Implementation of the gradient operator (∇), see
  * [[_root_.io.github.tailhq.dynaml.tensorflow.dynamics]],
  * for more details.
  * */
private[dynamics] class Gradient[D: TF: IsNotQuantized] extends SlicedGradient[D](name = s"Grad")(---)(---)

private[dynamics] object Gradient {
  def apply[D: TF : IsNotQuantized]: Gradient[D] = new Gradient()
}