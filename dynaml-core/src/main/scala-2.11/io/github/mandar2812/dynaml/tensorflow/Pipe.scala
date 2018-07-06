package io.github.mandar2812.dynaml.tensorflow

import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.tensorflow.utils.{GaussianScalerTF, MinMaxScalerTF}
import org.platanios.tensorflow.api._

private[tensorflow] object Pipe {

  def gauss_std(ax: Int = 0): DataPipe[Tensor, (Tensor, GaussianScalerTF)] =
    DataPipe((labels: Tensor) => {

      val labels_mean = labels.mean(axes = ax)

      val n_data = labels.shape(0).scalar.asInstanceOf[Int].toDouble

      val labels_sd = labels.subtract(labels_mean).square.mean(axes = ax).multiply(n_data/(n_data - 1d)).sqrt

      val labels_scaler = GaussianScalerTF(labels_mean, labels_sd)

      val labels_scaled = labels_scaler(labels)

      (labels_scaled, labels_scaler)
    })

  def minmax_std(ax: Int = 0): DataPipe[Tensor, (Tensor, MinMaxScalerTF)] =
    DataPipe((labels: Tensor) => {

      val labels_min = labels.min(axes = ax)
      val labels_max = labels.max(axes = ax)

      val labels_scaler = MinMaxScalerTF(labels_min, labels_max)

      val labels_scaled = labels_scaler(labels)

      (labels_scaled, labels_scaler)
    })

  val gaussian_standardization: DataPipe2[Tensor, Tensor, ((Tensor, Tensor), (GaussianScalerTF, GaussianScalerTF))] =
    DataPipe2((features: Tensor, labels: Tensor) => {

      val (features_mean, labels_mean) = (features.mean(axes = 0), labels.mean(axes = 0))

      val n_data = features.shape(0).scalar.asInstanceOf[Int].toDouble

      val (features_sd, labels_sd) = (
        features.subtract(features_mean).square.mean(axes = 0).multiply(n_data/(n_data - 1d)).sqrt,
        labels.subtract(labels_mean).square.mean(axes = 0).multiply(n_data/(n_data - 1d)).sqrt
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

  val minmax_standardization: DataPipe2[Tensor, Tensor, ((Tensor, Tensor), (MinMaxScalerTF, MinMaxScalerTF))] =
    DataPipe2((features: Tensor, labels: Tensor) => {

      val (features_min, labels_min) = (features.min(axes = 0), labels.min(axes = 0))

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

  val gauss_minmax_standardization: DataPipe2[Tensor, Tensor, ((Tensor, Tensor), (GaussianScalerTF, MinMaxScalerTF))] =
    DataPipe2((features: Tensor, labels: Tensor) => {

      val features_mean = features.mean(axes = 0)

      val (labels_min, labels_max) = (labels.min(axes = 0), labels.max(axes = 0))

      val n_data = features.shape(0).scalar.asInstanceOf[Int].toDouble

      val features_sd =
        features.subtract(features_mean).square.mean(axes = 0).multiply(n_data/(n_data - 1d)).sqrt

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
