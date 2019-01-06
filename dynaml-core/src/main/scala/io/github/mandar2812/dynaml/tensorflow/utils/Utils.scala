package io.github.mandar2812.dynaml.tensorflow.utils

import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe12}
import io.github.mandar2812.dynaml.tensorflow.data.AbstractDataSet
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.{Shape, Tensor}

object Utils {

  type NNPROP      = (Seq[Int], Seq[Shape], Seq[String], Seq[String])

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq(t: Tensor): Iterator[Double] = {
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
  def kl(prior: Output, p: Output): Output =
    prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

  def kl(prior: Tensor, p: Tensor): Output =
    prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

  /**
    * Calculate the Jensen Shannon divergence
    * between a probability and a target probability.
    * */
  def js(target_prob: Output, prob: Output) = {
    val m = target_prob.add(prob).divide(2.0)
    kl(target_prob, m).add(kl(prob, m)).multiply(0.5)
  }

  def js(target_prob: Tensor, prob: Tensor) = {
    val m = target_prob.add(prob).divide(2.0)
    kl(target_prob, m).add(kl(prob, m)).multiply(0.5)
  }

  /**
    * Calculate the Hellinger distance between two
    * probability distributions.
    * */
  def hellinger(target_prob: Output, prob: Output) =
    target_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))

  def hellinger(target_prob: Tensor, prob: Tensor) =
    target_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))

  def cross_entropy(target_prob: Output, prob: Output) =
    target_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()

  /**
    * Calculate the cross-entropy of two
    * probability distributions.
    * */
  def cross_entropy(target_prob: Tensor, prob: Tensor) =
    target_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()

  def buffered_preds[
  IT, IO, ID, IS, I,
  TT, TO, TD, TS, EI,
  InferInput, InferOutput,
  ModelInferenceOutput](
    predictiveModel: Estimator[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS), (I, EI)],
    workingData: InferInput,
    buffer: Int, dataSize: Int)(
    implicit getSplitByIndex: MetaPipe12[InferInput, Int, Int, InferInput],
    concatenateSplits: DataPipe[Iterable[InferOutput], InferOutput],
    evFetchableIO: Fetchable.Aux[IO, IT],
    evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
    ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): InferOutput = {

    val get_data_split = getSplitByIndex(workingData)

    val preds_splits: Iterable[InferOutput] = (0 until dataSize)
      .grouped(buffer)
      .map(indices => {

        val progress = math.round(10*indices.head*buffer*100.0/dataSize)/10d

        print("Progress %:\t")
        pprint.pprintln(progress)

        predictiveModel.infer[InferInput, InferOutput, ModelInferenceOutput](
          () => get_data_split(indices.head, indices.last))
      }).toIterable

    concatenateSplits(preds_splits)
  }

  def predict_data[
  IT, IO, ID, IS, I,
  TT, TO, TD, TS, EI,
  InferOutput, ModelInferenceOutput](
    predictiveModel: Estimator[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS), (I, EI)],
    data: AbstractDataSet[IT, TT],
    pred_flags: (Boolean, Boolean) = (false, true),
    buff_size: Int = 400)(
    implicit getSplitByIndex: MetaPipe12[IT, Int, Int, IT],
    concatenateSplits: DataPipe[Iterable[InferOutput], InferOutput],
    evFetchableIO: Fetchable.Aux[IO, IT],
    evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
    evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
    ev: Estimator.SupportedInferInput[IT, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): (Option[InferOutput], Option[InferOutput]) = {

    val train_preds: Option[InferOutput] =
      if (pred_flags._1) {

        println("\nGenerating predictions for training data.\n")

        val predictions = buffered_preds[
          IT, IO, ID, IS, I,
          TT, TO, TD, TS, EI,
          IT, InferOutput,
          ModelInferenceOutput](
          predictiveModel,
          data.trainData,
          buff_size,
          data.nTrain)

        Some(predictions)

      } else None

    val test_preds: Option[InferOutput] =
      if (pred_flags._2) {

        println("\nGenerating predictions for test data.\n")

        val predictions = buffered_preds[
          IT, IO, ID, IS, I,
          TT, TO, TD, TS, EI,
          IT, InferOutput,
          ModelInferenceOutput](
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
