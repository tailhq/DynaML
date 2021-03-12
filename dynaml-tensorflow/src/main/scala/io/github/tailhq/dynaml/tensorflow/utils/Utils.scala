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
package io.github.tailhq.dynaml.tensorflow.utils

import io.github.tailhq.dynaml.pipes.{DataPipe, MetaPipe12}
import io.github.tailhq.dynaml.tensorflow.data.AbstractDataSet
import io.github.tailhq.dynaml.tensorflow.Learn
import io.github.tailhq.dynaml.tensorflow.layers._
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, IsReal, IsNotQuantized, TF}
import org.platanios.tensorflow.api.{Tensor, Shape, Output, tfi, tf}
import org.platanios.tensorflow.api.implicits.helpers.{OutputToDataType, OutputToShape, OutputToTensor}
import org.platanios.tensorflow.api.learn.layers.{Compose, Concatenate, Layer, Linear, Conv2D, Map => MapTF, MapSeq}



object Utils {

  type NNPROP      = (Seq[Int], Seq[Shape], Seq[String], Seq[String])

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq[D : TF: IsReal](t: Tensor[D]): Iterator[Double] = t.castTo[Double].entriesIterator



  def process_scope(s: String): String = if(s.isEmpty) "" else s"$s/"

  /**
    * Find out the name scope of a layer which is part
    * of a larger architecture.
    *
    * @param architecture A Neural network architecture
    * @param layer_name The constituent layer to search for
    * @return The name scope for the layer
    * */
  def get_scope(
    architecture: Layer[_, _])(
    layer_name: String): String = {

    def scope_search(lstack: Seq[Layer[_, _]], scopesAcc: Seq[String]): String = lstack match {

      case Seq() => scopesAcc.headOption.getOrElse("")

      case Linear(name, _, _, _, _) :: tail =>
        if(name == layer_name) s"${scopesAcc.head}" else scope_search(tail, scopesAcc.tail)

      case Conv2D(name, _, _, _, _, _, _, _, _) :: tail =>
        if(name == layer_name) s"${scopesAcc.head}" else scope_search(tail, scopesAcc.tail)

      case FiniteHorizonCTRNN(name, _, _, _, _, _, _, _) :: tail =>
        if(name == layer_name) s"${scopesAcc.head}" else scope_search(tail, scopesAcc.tail)

      case FiniteHorizonLinear(name, _, _, _, _) :: tail =>
        if(name == layer_name) s"${scopesAcc.head}" else scope_search(tail, scopesAcc.tail)

      case DynamicTimeStepCTRNN(name, _, _, _, _, _, _) :: tail =>
        if(name == layer_name) s"${scopesAcc.head}" else scope_search(tail, scopesAcc.tail)

      case Compose(name, l1, l2) :: tail =>
        scope_search(Seq(l1, l2) ++ tail, Seq.fill(2)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case Concatenate(name, ls) :: tail =>
        scope_search(ls ++ tail, Seq.fill(ls.length)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case MapTF(name, ls, _) :: tail =>
        scope_search(Seq(ls) ++ tail, Seq(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case MapSeq(name, ls, l) :: tail =>
        scope_search((ls :: tail), Seq.fill(2)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case SeqLayer(name, ls) :: tail =>
        scope_search(ls ++ tail, Seq.fill(ls.length)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case ArrayLayer(name, ls) :: tail =>
        scope_search(ls ++ tail, Seq.fill(ls.length)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case CombinedLayer(name, ls) :: tail =>
        scope_search(ls ++ tail, Seq.fill(ls.length)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case CombinedArrayLayer(name, ls) :: tail =>
        scope_search(ls ++ tail, Seq.fill(ls.length)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case MapLayer(name, ls) :: tail =>
        scope_search(ls.values.toSeq ++ tail, Seq.fill(ls.size)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      //TODO: Need to test this part!!
      case ScopedMapLayer(name, ls, scopes) :: tail =>
        scope_search(ls.values.toSeq ++ tail, scopes ++ scopesAcc.tail)

      case BifurcationLayer(name, l1, l2) :: tail =>
        scope_search(Seq(l1, l2) ++ tail, Seq.fill(2)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case Tuple2Layer(name, l1, l2) :: tail =>
        scope_search(Seq(l1, l2) ++ tail, Seq.fill(2)(s"${process_scope(scopesAcc.head)}$name") ++ scopesAcc.tail)

      case head :: tail =>
        if(head.name == layer_name) s"${scopesAcc.head}" else scope_search(tail, scopesAcc.tail)
    }

    scope_search(Seq(architecture), Seq(""))
  }


  /**
    * Returns the properties [[NNPROP]] (i.e. layer sizes, shapes, parameter names, & data types)
    * of a feed-forward/dense neural stack which consists of layers of unequal size.
    *
    * @param d The dimensionality of the input (assumed to be a rank 1 tensor).
    * @param num_pred_dims The dimensionality of the network output.
    * @param layer_sizes The size of each hidden layer.
    * @param dType The data type of the layer weights and biases.
    * @param starting_index The numeric index of the first layer, defaults to 1.
    *
    * */
  def get_ffstack_properties(
    d: Int, num_pred_dims: Int,
    layer_sizes: Seq[Int],
    dType: String = "FLOAT64",
    starting_index: Int = 1): NNPROP = {

    val net_layer_sizes       = Seq(d) ++ layer_sizes ++ Seq(num_pred_dims)

    val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))

    val size                  = net_layer_sizes.tail.length

    val layer_parameter_names = (starting_index until starting_index + size).map(i => s"Linear_$i/Weights")

    val layer_datatypes       = Seq.fill(net_layer_sizes.tail.length)(dType)

    (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes)
  }

  /**
    * Calculate the Kullback Leibler divergence of
    * a probability density from a prior density.
    * */
  def kl[D : TF: IsNotQuantized](prior: Output[D], p: Output[D]): Output[D] =
    prior.divide(p).log.multiply(prior).sum(axes = 1).mean[Int]()

  def kl[D : TF: IsNotQuantized](prior: Tensor[D], p: Tensor[D]): Tensor[D] =
    prior.divide(p).log.multiply(prior).sum(axes = 1).mean[Int]()

  /**
    * Calculate the Jensen Shannon divergence
    * between a probability and a target probability.
    * */
  def js[D : TF: IsNotQuantized](target_prob: Output[D], prob: Output[D]): Output[D] = {
    val m = target_prob.add(prob).divide(Tensor(2.0).toOutput.castTo[D])
    kl(target_prob, m).add(kl(prob, m)).multiply(Tensor(0.5).toOutput.castTo[D])
  }

  def js[D : TF: IsNotQuantized](target_prob: Tensor[D], prob: Tensor[D]): Tensor[D] = {
    val two = Tensor(2).castTo[D]
    val m = tfi.divide[D](target_prob.add(prob), two)
    tfi.divide[D](kl(target_prob, m).add(kl(prob, m)), two)
  }

  /**
    * Calculate the Hellinger distance between two
    * probability distributions.
    * */
  def hellinger[D : TF: IsNotQuantized](target_prob: Output[D], prob: Output[D]): Output[D] =
    target_prob.sqrt.subtract(prob.sqrt).square.sum(axes = 1).sqrt.divide(Tensor(math.sqrt(2.0)).toOutput.castTo[D])

  def hellinger[D : TF: IsNotQuantized](target_prob: Tensor[D], prob: Tensor[D]): Tensor[D] =
    target_prob.sqrt.subtract(prob.sqrt).square.sum(axes = 1).sqrt

  def cross_entropy[D : TF: IsNotQuantized](target_prob: Output[D], prob: Output[D]): Output[D] =
    target_prob.multiply(prob.log).sum(axes = 1).multiply(Tensor(-1.0).toOutput.castTo[D]).mean[Int]()

  /**
    * Calculate the cross-entropy of two
    * probability distributions.
    * */
  def cross_entropy[D : TF: IsNotQuantized](target_prob: Tensor[D], prob: Tensor[D]): Output[D] =
    target_prob.multiply(prob.log).sum(axes = 1).multiply(Tensor(-1.0).castTo(target_prob.dataType)).mean[Int]()

  def buffered_preds[
  In, TrainIn, TrainOut, Out,
  Loss: TF : IsFloatOrDouble,
  IT, ID, IS, ITT, ITD, ITS,
  TT, InferIn, InferOut](
    predictiveModel: Learn.SupEstimatorTF[In, TrainIn, TrainOut, Out, Loss, (Out, (In, TrainIn))],
    workingData: InferIn,
    buffer: Int, dataSize: Int)(
    implicit
    getSplitByIndex: MetaPipe12[InferIn, Int, Int, InferIn],
    concatenateSplits: DataPipe[Iterable[InferOut], InferOut],
    evOutputToDataTypeIn: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeOut: OutputToDataType.Aux[Out, ITD],
    evOutputToShapeIn: OutputToShape.Aux[In, IS],
    evOutputToShapeOut: OutputToShape.Aux[Out, ITS],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    evOutputToTensorOut: OutputToTensor.Aux[Out, ITT],
    ev: Estimator.SupportedInferInput[In, IT, ITT, InferIn, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, Out), (IT, ITT)]
  ): InferOut = {

    val get_data_split = getSplitByIndex(workingData)

    val preds_splits: Iterable[InferOut] = (0 until dataSize)
      .grouped(buffer)
      .map(indices => {

        val progress = math.round(10*indices.head*buffer*100.0/dataSize)/10d

        print("Progress %:\t")
        pprint.pprintln(progress)

        predictiveModel.infer[IT, ID, IS, ITT, ITD, ITS, InferIn, InferOut](
          () => get_data_split(indices.head, indices.last))
      }).toIterable

    concatenateSplits(preds_splits)
  }

  def predict_data[
  In, TrainIn, TrainOut, Out,
  Loss: TF : IsFloatOrDouble,
  IT, ID, IS, ITT, ITD, ITS,
  TT, InferOut](
    predictiveModel: Learn.SupEstimatorTF[In, TrainIn, TrainOut, Out, Loss, (Out, (In, TrainIn))],
    data: AbstractDataSet[IT, TT],
    pred_flags: (Boolean, Boolean) = (false, true),
    buff_size: Int = 400)(
    implicit
    getSplitByIndex: MetaPipe12[IT, Int, Int, IT],
    concatenateSplits: DataPipe[Iterable[InferOut], InferOut],
    evOutputToDataTypeIn: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeOut: OutputToDataType.Aux[Out, ITD],
    evOutputToShapeIn: OutputToShape.Aux[In, IS],
    evOutputToShapeOut: OutputToShape.Aux[Out, ITS],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    evOutputToTensorOut: OutputToTensor.Aux[Out, ITT],
    ev: Estimator.SupportedInferInput[In, IT, ITT, IT, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, Out), (IT, ITT)]): (Option[InferOut], Option[InferOut]) = {

    val train_preds: Option[InferOut] =
      if (pred_flags._1) {

        println("\nGenerating predictions for training data.\n")

        val predictions = buffered_preds[
          In, TrainIn, TrainOut, Out, Loss,
          IT, ID, IS, ITT, ITD, ITS,
          TT, IT, InferOut](
          predictiveModel,
          data.trainData,
          buff_size,
          data.nTrain)

        Some(predictions)

      } else None

    val test_preds: Option[InferOut] =
      if (pred_flags._2) {

        println("\nGenerating predictions for test data.\n")

        val predictions = buffered_preds[
          In, TrainIn, TrainOut, Out, Loss,
          IT, ID, IS, ITT, ITD, ITS,
          TT, IT, InferOut](
          predictiveModel,
          data.testData,
          buff_size,
          data.nTest)

        Some(predictions)
      } else None

    (train_preds, test_preds)
  }


}
