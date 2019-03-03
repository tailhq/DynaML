package io.github.mandar2812.dynaml.tensorflow.utils

import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe12}
import io.github.mandar2812.dynaml.tensorflow.data.AbstractDataSet
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, IsNotQuantized, TF}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputToDataType, OutputToShape, OutputToTensor}


object Utils {

  type NNPROP      = (Seq[Int], Seq[Shape], Seq[String], Seq[String])

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq[D : TF: IsNotQuantized](t: Tensor[D]): Iterator[Double] = {
    val datatype = t.dataType.toString()
    t.entriesIterator.map(x =>
      if(datatype == "FLOAT64") x.asInstanceOf[Double]
      else x.asInstanceOf[Float].toDouble)
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
  IT, ID, IS, TT, TD, TS,
  InferIn, InferOut](
    predictiveModel: Estimator[In, TrainIn, TrainOut, Out, Loss, (Out, TrainOut)],
    workingData: InferIn,
    buffer: Int, dataSize: Int)(
    implicit
    getSplitByIndex: MetaPipe12[InferIn, Int, Int, InferIn],
    concatenateSplits: DataPipe[Iterable[InferOut], InferOut],
    evOutputToDataTypeIn: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeOut: OutputToDataType.Aux[TrainOut, TD],
    evOutputToShapeIn: OutputToShape.Aux[In, IS],
    evOutputToShapeOut: OutputToShape.Aux[TrainOut, TS],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    evOutputToTensorOut: OutputToTensor.Aux[TrainOut, TT],
    ev: Estimator.SupportedInferInput[In, IT, TT, InferIn, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, Out), (IT, TT)]
  ): InferOut = {

    val get_data_split = getSplitByIndex(workingData)

    val preds_splits: Iterable[InferOut] = (0 until dataSize)
      .grouped(buffer)
      .map(indices => {

        val progress = math.round(10*indices.head*buffer*100.0/dataSize)/10d

        print("Progress %:\t")
        pprint.pprintln(progress)

        predictiveModel.infer[IT, ID, IS, TT, TD, TS, InferIn, InferOut](
          () => get_data_split(indices.head, indices.last))
      }).toIterable

    concatenateSplits(preds_splits)
  }

  def predict_data[
  In, TrainIn, TrainOut, Out,
  Loss: TF : IsFloatOrDouble,
  IT, ID, IS, TT, TD, TS,
  InferOut](
    predictiveModel: Estimator[In, TrainIn, TrainOut, Out, Loss, (Out, TrainOut)],
    data: AbstractDataSet[IT, TT],
    pred_flags: (Boolean, Boolean) = (false, true),
    buff_size: Int = 400)(
    implicit
    getSplitByIndex: MetaPipe12[IT, Int, Int, IT],
    concatenateSplits: DataPipe[Iterable[InferOut], InferOut],
    evOutputToDataTypeIn: OutputToDataType.Aux[In, ID],
    evOutputToDataTypeOut: OutputToDataType.Aux[TrainOut, TD],
    evOutputToShapeIn: OutputToShape.Aux[In, IS],
    evOutputToShapeOut: OutputToShape.Aux[TrainOut, TS],
    evOutputToTensorIn: OutputToTensor.Aux[In, IT],
    evOutputToTensorOut: OutputToTensor.Aux[TrainOut, TT],
    ev: Estimator.SupportedInferInput[In, IT, TT, IT, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evOutputToTensorInOut: OutputToTensor.Aux[(In, Out), (IT, TT)]): (Option[InferOut], Option[InferOut]) = {

    val train_preds: Option[InferOut] =
      if (pred_flags._1) {

        println("\nGenerating predictions for training data.\n")

        val predictions = buffered_preds[
          In, TrainIn, TrainOut, Out, Loss,
          IT, ID, IS, TT, TD, TS,
          IT, InferOut](
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
          IT, ID, IS, TT, TD, TS,
          IT, InferOut](
          predictiveModel,
          data.testData,
          buff_size,
          data.nTest)

        Some(predictions)
      } else None

    (train_preds, test_preds)
  }

  /*def predict[
  IT, IO, ID, IS, I,
  TT, TO, TD, TS, EI,
  ModelInferenceOutput](
    predictiveModel: Estimator[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS), (I, EI)],
    data: DataSet[IT, IO, ID, IS])(
    implicit evFetchableIO: Fetchable.Aux[IO, IT],
    evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
    ev: Estimator.SupportedInferInput[
      Dataset[IT,IO,ID,IS], Iterator[(IT, ModelInferenceOutput)],
      IT, IO, ID, IS, ModelInferenceOutput]
  ): Iterator[(IT, ModelInferenceOutput)] = predictiveModel.infer(() => data.get)*/

}
