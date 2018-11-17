package io.github.mandar2812.dynaml.tensorflow.utils

import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe12}
import io.github.mandar2812.dynaml.tensorflow.data.AbstractDataSet
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.core.types.{IsFloat32OrFloat64, IsNotQuantized, TF}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure


object Utils {

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq[D : TF: IsNotQuantized](t: Tensor[D]): Iterator[Double] = {
    val datatype = t.dataType.toString()
    t.entriesIterator.map(x =>
      if(datatype == "FLOAT64") x.asInstanceOf[Double]
      else x.asInstanceOf[Float].toDouble)
  }


  def get_ffstack_properties(
    neuron_counts: Seq[Int],
    ff_index: Int,
    data_type: String): (Seq[Shape], Seq[String], Seq[String]) = {

    val layer_parameter_names = (ff_index until ff_index + neuron_counts.length - 1).map(i => "Linear_"+i+"/Weights")
    val layer_shapes          = neuron_counts.sliding(2).toSeq.map(c => Shape(c.head, c.last))
    val layer_datatypes       = Seq.fill(layer_shapes.length)(data_type)


    (layer_shapes, layer_parameter_names, layer_datatypes)
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
  Loss: TF : IsFloat32OrFloat64,
  IT, ID, IS, TT, TD, TS,
  InferIn, InferOut](
    predictiveModel: Estimator[In, TrainIn, TrainOut, Out, Loss, (Out, TrainOut)],
    workingData: InferIn,
    buffer: Int, dataSize: Int)(
    implicit getSplitByIndex: MetaPipe12[InferIn, Int, Int, InferIn],
    concatenateSplits: DataPipe[Iterable[InferOut], InferOut],
    evFetchableIn: NestedStructure.Aux[In, IT, ID, IS],
    evFetchableOut: NestedStructure.Aux[Out, TT, TD, TS],
    evFetchableTOut: NestedStructure.Aux[TrainOut, TT, TD, TS],
    ev: Estimator.SupportedInferInput[In, IT, TT, InferIn, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evFetchableIO: NestedStructure.Aux[(In, TrainOut), (IT, TT), (ID, TD), (IS, TS)],
    evFetchableInOut: NestedStructure.Aux[(In, Out), (IT, TT), (ID, TD), (IS, TS)]
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
  Loss: TF : IsFloat32OrFloat64,
  IT, ID, IS, TT, TD, TS,
  InferOut](
    predictiveModel: Estimator[In, TrainIn, TrainOut, Out, Loss, (Out, TrainOut)],
    data: AbstractDataSet[IT, TT],
    pred_flags: (Boolean, Boolean) = (false, true),
    buff_size: Int = 400)(
    implicit getSplitByIndex: MetaPipe12[IT, Int, Int, IT],
    concatenateSplits: DataPipe[Iterable[InferOut], InferOut],
    evFetchableIn: NestedStructure.Aux[In, IT, ID, IS],
    evFetchableOut: NestedStructure.Aux[Out, TT, TD, TS],
    evFetchableTOut: NestedStructure.Aux[TrainOut, TT, TD, TS],
    ev: Estimator.SupportedInferInput[In, IT, TT, IT, InferOut],
    // This implicit helps the Scala 2.11 compiler.
    evFetchableIO: NestedStructure.Aux[(In, TrainOut), (IT, TT), (ID, TD), (IS, TS)],
    evFetchableInOut: NestedStructure.Aux[(In, Out), (IT, TT), (ID, TD), (IS, TS)]
  ): (Option[InferOut], Option[InferOut]) = {

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
