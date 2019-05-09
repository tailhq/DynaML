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
package io.github.mandar2812.dynaml.models

import ammonite.ops._
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.tensorflow._
import org.scalatest.{FlatSpec, Matchers}
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Layer

class TFModelSpec extends FlatSpec with Matchers {
  "DynaML TensorFlow model wrappers" should " train and predict as expected" in {

    val tempdir = home / "tmp"

    val summary_dir = tempdir / s"dtf_model_test-${DateTime.now().toString("YYYY-MM-dd-HH-mm")}"

    val (weight, bias) = (2.5, 1.5)

    val data_size = 100
    val rv        = GaussianRV(0.0, 2.0).iid(data_size)

    val data = dtfdata
      .dataset(rv.draw)
      .to_supervised(
        DataPipe[Double, (Tensor[Double], Tensor[Double])](
          n => (dtf.tensor_f64(1)(n), dtf.tensor_f64(1)(n * weight + bias))
        )
      )

    val test_data = dtfdata
      .dataset(rv.draw)
      .to_supervised(
        DataPipe[Double, (Tensor[Double], Tensor[Double])](
          n => (dtf.tensor_f64(1)(n), dtf.tensor_f64(1)(n * weight + bias))
        )
      )

    val train_fraction = 0.7

    val tf_dataset = data.partition(
      DataPipe[(Tensor[Double], Tensor[Double]), Boolean](
        _ => scala.util.Random.nextDouble() <= train_fraction
      )
    )

    val arch: Layer[Output[Double], Output[Double]] =
      dtflearn.feedforward[Double](num_units = 1)(id = 1)

    val loss: Layer[(Output[Double], Output[Double]), Output[Double]] =
      tf.learn.L2Loss[Double, Double]("Loss/L2") >>
        tf.learn.Mean[Double]("Loss/Mean") >>
        tf.learn.ScalarSummary[Double]("Loss/ModelLoss", "ModelLoss")

    val graphInstance = Graph()

    val regression_model =
      dtflearn.model[Output[Double], Output[Double], Output[
        Double
      ], Double, Tensor[Double], FLOAT64, Shape, Tensor[Double], FLOAT64, Shape, Tensor[
        Double
      ], FLOAT64, Shape](
        arch,
        (FLOAT64, Shape(1)),
        (FLOAT64, Shape(1)),
        loss,
        existingGraph = Some(graphInstance)
      )

    val train_config = dtflearn.model.trainConfig(
      summary_dir,
      dtflearn.model.data_ops[(Output[Double], Output[Double])](
        shuffleBuffer = 5000,
        batchSize = 16,
        prefetchSize = 10
      ),
      tf.train.Adam(0.1f),
      dtflearn.rel_loss_change_stop(0.05, 5000),
      Some(
        dtflearn.model._train_hooks(
          summary_dir,
          stepRateFreq = 1000,
          summarySaveFreq = 1000,
          checkPointFreq = 1000
        )
      )
    )

    val pattern_to_tensor = DataPipe(
      (ds: Seq[(Tensor[Double], Tensor[Double])]) => {
        val (xs, ys) = ds.unzip

        (
          dtfpipe.EagerStack[Double](axis = 0).run(xs),
          dtfpipe.EagerStack[Double](axis = 0).run(ys)
        )
      }
    )

    val pattern_to_sym = 
      DataPipe(
        (ds: Seq[(Tensor[Double], Tensor[Double])]) => {
          val (xs, ys)   = ds.unzip
          val batch_size = ds.length
          val (xt, yt) = (
            dtfpipe.LazyStack[Double](axis = 0).run(xs.map(_.toOutput)),
            dtfpipe.LazyStack[Double](axis = 0).run(ys.map(_.toOutput))
          )

          (
            xt,
            yt
          )

        }
      )

    regression_model.train(
      tf_dataset.training_dataset,
      train_config,
      dtflearn.model.tf_data_handle_ops(
        bufferSize = 500,
        //patternToSym = Some(pattern_to_sym),
        patternToTensor = Some(pattern_to_tensor),
        concatOpO = Some(dtfpipe.EagerConcatenate[Double]())
      )
    )

    val test_pred =
      regression_model.predict(Tensor[Double](1.0d).reshape(Shape(1, 1))).scalar

    assert(test_pred == 4.0)

    val epsilon = 1e-5

    val metrics = regression_model.evaluate(
      test_data,
      Seq(
        dtflearn.mse[Output[Double], Double](),
        dtflearn.mae[Output[Double], Double]()
      ),
      dtflearn.model.data_ops[(Output[Double], Output[Double])](
        repeat = 0,
        shuffleBuffer = 0,
        batchSize = 16,
        prefetchSize = 10
      ),
      dtflearn.model.tf_data_handle_ops(
        bufferSize = 500,
        patternToTensor = Some(pattern_to_tensor),
        concatOpO = Some(dtfpipe.EagerConcatenate[Double]())
      )
    )

    assert(metrics.forall(m => m.scalar.toDouble <= epsilon))

  }
}
