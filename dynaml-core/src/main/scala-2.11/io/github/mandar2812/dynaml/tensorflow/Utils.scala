package io.github.mandar2812.dynaml.tensorflow

import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe12}
import io.github.mandar2812.dynaml.tensorflow.utils.AbstractDataSet
import org.platanios.tensorflow.api.{Shape, Tensor}
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.Output

object Utils {

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq(t: Tensor): Iterator[Double] = {
    val datatype = t.dataType.toString()
    t.entriesIterator.map(x =>
      if(datatype == "FLOAT64") x.asInstanceOf[Double]
      else x.asInstanceOf[Float].toDouble)
  }


  def get_ffstack_properties(
    neuron_counts: Seq[Int],
    ff_index: Int): (Seq[Shape], Seq[String], Seq[String]) = {

    val layer_parameter_names = (ff_index until ff_index + neuron_counts.length - 1).map(i => "Linear_"+i+"/Weights")
    val layer_shapes          = neuron_counts.sliding(2).toSeq.map(c => Shape(c.head, c.last))
    val layer_datatypes       = Seq.fill(layer_shapes.length)("FLOAT64")


    (layer_shapes, layer_parameter_names, layer_datatypes)
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
      })
      .toIterable

    concatenateSplits(preds_splits)
  }

  def predict_data[
  IT, IO, ID, IS, I,
  TT, TO, TD, TS, EI,
  InferOutput,
  ModelInferenceOutput](
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

}
