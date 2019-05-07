package io.github.mandar2812.dynaml.tensorflow

import ammonite.ops.Path
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.tensorflow.utils.{
  GaussianScalerTF,
  MinMaxScalerTF
}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{
  IsNotQuantized,
  IsReal,
  TF,
  UByte
}
import org.platanios.tensorflow.api.ops.Image.DCTMethod

private[tensorflow] object Pipe {

  abstract class TFOpsPipe[T: TF]
      extends DataPipe[Iterable[Tensor[T]], Tensor[T]] {
    self =>
    def zip[J: TF](other: TFOpsPipe[J]): Tuple2OpsPipe[T, J] =
      Tuple2OpsPipe(self, other)
  }

  case class Tuple2OpsPipe[I: TF, J: TF](
    pipe1: TFOpsPipe[I],
    pipe2: TFOpsPipe[J])
      extends DataPipe[Iterable[(Tensor[I], Tensor[J])], (Tensor[I], Tensor[J])] {

    override def run(
      x: Iterable[(Tensor[I], Tensor[J])]
    ): (Tensor[I], Tensor[J]) = {
      val (xs, ys) = x.unzip

      (pipe1(xs), pipe2(ys))
    }
  }

  abstract class TFSymOpsPipe[T: TF]
      extends DataPipe[Iterable[Output[T]], Output[T]] {

    self =>
    def zip[J: TF](other: TFSymOpsPipe[J]): Tuple2SymOpsPipe[T, J] =
      Tuple2SymOpsPipe(self, other)
  }

  case class Tuple2SymOpsPipe[I: TF, J: TF](
    pipe1: TFSymOpsPipe[I],
    pipe2: TFSymOpsPipe[J])
      extends DataPipe[Iterable[(Output[I], Output[J])], (Output[I], Output[J])] {

    override def run(
      x: Iterable[(Output[I], Output[J])]
    ): (Output[I], Output[J]) = {
      val (xs, ys) = x.unzip

      (pipe1(xs), pipe2(ys))
    }
  }
  case class EagerStack[T: TF](axis: Int = 0) extends TFOpsPipe[T] {
    override def run(data: Iterable[Tensor[T]]): Tensor[T] =
      tfi.stack(data.toSeq, axis)
  }

  case class EagerConcatenate[T: TF](axis: Int = 0) extends TFOpsPipe[T] {
    override def run(data: Iterable[Tensor[T]]): Tensor[T] =
      tfi.concatenate(data.toSeq, axis)
  }

  case class LazyStack[T: TF](axis: Int = 0) extends TFSymOpsPipe[T] {
    override def run(data: Iterable[Output[T]]): Output[T] =
      tf.stack(data.toSeq, axis)
  }

  case class LazyConcatenate[T: TF](axis: Int = 0) extends TFSymOpsPipe[T] {
    override def run(data: Iterable[Output[T]]): Output[T] =
      tf.concatenate(data.toSeq, axis)
  }

  /**
    * Read a file into a String Tensor
    * */
  def read_image(
    num_channels: Int = 0,
    ratio: Int = 1,
    fancyUpscaling: Boolean = true,
    tryRecoverTruncated: Boolean = false,
    acceptableFraction: Float = 1,
    dctMethod: DCTMethod = DCTMethod.SystemDefault,
    name: String = "DecodeJpeg"
  ): DataPipe[Path, Output[UByte]] =
    DataPipe((image: Path) => Tensor(image.toString()).toOutput) >
      DataPipe(
        (content: Output[String]) =>
          tf.image.decodeJpeg(
            content,
            num_channels,
            ratio,
            fancyUpscaling,
            tryRecoverTruncated,
            acceptableFraction,
            dctMethod,
            name
          )
      )

  def resize_image[T: TF: IsReal](
    size: (Int, Int),
    alignCorners: Boolean = false,
    name: String = "ResizeArea"
  ): DataPipe[Output[T], Output[Float]] =
    DataPipe(
      (image_content: Output[T]) =>
        tf.image.resizeArea(
          image_content,
          Array(size._1, size._2),
          alignCorners,
          name
        )
    )

  def extract_image_patch[T: TF](
    heightRange: Range,
    widthRange: Range
  ): DataPipe[Output[T], Output[T]] =
    DataPipe(
      (image: Output[T]) =>
        image(
          ---,
          heightRange.min :: heightRange.max + 1,
          widthRange.min :: widthRange.max + 1,
          ::
        )
    )

  def gauss_std[D: TF: IsNotQuantized](
    ax: Int = 0
  ): DataPipe[Tensor[D], (Tensor[D], GaussianScalerTF[D])] =
    DataPipe((labels: Tensor[D]) => {

      val labels_mean: Tensor[D] = labels.mean(axes = Tensor(ax))

      val n_data: Double = labels.shape(0).scalar.toDouble

      val labels_sd: Tensor[D] = tfi
        .multiply[D](
          labels.subtract(labels_mean).square.mean(axes = ax),
          Tensor(n_data / (n_data - 1d)).castTo[D]
        )
        .sqrt

      val labels_scaler = GaussianScalerTF(labels_mean, labels_sd)

      val labels_scaled = labels_scaler(labels)

      (labels_scaled, labels_scaler)
    })

  def minmax_std[D: TF: IsNotQuantized](
    ax: Int = 0
  ): DataPipe[Tensor[D], (Tensor[D], MinMaxScalerTF[D])] =
    DataPipe((labels: Tensor[D]) => {

      val labels_min = labels.min(axes = ax)
      val labels_max = labels.max(axes = ax)

      val labels_scaler = MinMaxScalerTF(labels_min, labels_max)

      val labels_scaled = labels_scaler(labels)

      (labels_scaled, labels_scaler)
    })

  def gaussian_standardization[
    D: TF: IsNotQuantized,
    E: TF: IsNotQuantized
  ]: DataPipe2[Tensor[D], Tensor[
    E
  ], ((Tensor[D], Tensor[E]), (GaussianScalerTF[D], GaussianScalerTF[E]))] =
    DataPipe2((features: Tensor[D], labels: Tensor[E]) => {

      val (features_mean, labels_mean) =
        (features.mean(axes = 0), labels.mean(axes = 0))

      val n_data = features.shape(0).scalar.toDouble

      val (features_sd, labels_sd) = (
        tfi
          .multiply[D](
            features.subtract(features_mean).square.mean(axes = 0),
            Tensor(n_data / (n_data - 1d)).castTo[D]
          )
          .sqrt,
        tfi
          .multiply[E](
            labels.subtract(labels_mean).square.mean(axes = 0),
            Tensor(n_data / (n_data - 1d)).castTo[E]
          )
          .sqrt
      )

      val (features_scaler, labels_scaler) = (
        GaussianScalerTF(features_mean, features_sd),
        GaussianScalerTF(labels_mean, labels_sd)
      )

      val (features_scaled, labels_scaled) = (
        features_scaler(features),
        labels_scaler(labels)
      )

      ((features_scaled, labels_scaled), (features_scaler, labels_scaler))
    })

  def minmax_standardization[
    D: TF: IsNotQuantized,
    E: TF: IsNotQuantized
  ]: DataPipe2[Tensor[
    D
  ], Tensor[E], ((Tensor[D], Tensor[E]), (MinMaxScalerTF[D], MinMaxScalerTF[E]))] =
    DataPipe2((features: Tensor[D], labels: Tensor[E]) => {

      val (features_min, labels_min) =
        (features.min(axes = 0), labels.min(axes = 0))

      val (features_max, labels_max) = (
        features.max(axes = 0),
        labels.max(axes = 0)
      )

      val (features_scaler, labels_scaler) = (
        MinMaxScalerTF(features_min, features_max),
        MinMaxScalerTF(labels_min, labels_max)
      )

      val (features_scaled, labels_scaled) = (
        features_scaler(features),
        labels_scaler(labels)
      )

      ((features_scaled, labels_scaled), (features_scaler, labels_scaler))
    })

  def gauss_minmax_standardization[
    D: TF: IsNotQuantized,
    E: TF: IsNotQuantized
  ]: DataPipe2[Tensor[
    D
  ], Tensor[E], ((Tensor[D], Tensor[E]), (GaussianScalerTF[D], MinMaxScalerTF[E]))] =
    DataPipe2((features: Tensor[D], labels: Tensor[E]) => {

      val features_mean = features.mean(axes = 0)

      val (labels_min, labels_max) =
        (labels.min(axes = 0), labels.max(axes = 0))

      val n_data = features.shape(0).scalar.toDouble

      val features_sd = tfi
        .multiply[D](
          features.subtract(features_mean).square.mean(axes = 0),
          Tensor(n_data / (n_data - 1d)).castTo[D]
        )
        .sqrt

      val (features_scaler, labels_scaler) = (
        GaussianScalerTF(features_mean, features_sd),
        MinMaxScalerTF(labels_min, labels_max)
      )

      val (features_scaled, labels_scaled) = (
        features_scaler(features),
        labels_scaler(labels)
      )

      ((features_scaled, labels_scaled), (features_scaler, labels_scaler))
    })

}
