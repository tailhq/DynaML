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
package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseVector => BDV}
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.BinaryClassificationMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.kernels.LocalSVMKernel
import io.github.mandar2812.dynaml.modelpipe.GLMPipe
import io.github.mandar2812.dynaml.models.lm.{
  GeneralizedLinearModel,
  LogisticGLM,
  ProbitGLM
}
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.models.svm.DLSSVM
import io.github.mandar2812.dynaml.pipes._

/**
  * Created by mandar on 11/1/16.
  */
object TestNNWineQuality {
  def apply(
    hidden: Int = 2,
    nCounts: List[Int] = List(),
    acts: List[String],
    training: Int = 100,
    test: Int = 1000,
    columns: List[Int] = List(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    stepSize: Double = 0.01,
    maxIt: Int = 30,
    mini: Double = 1.0,
    alpha: Double = 0.5,
    regularization: Double = 0.5,
    wineType: String = "red"
  ): Unit = {

    //Load wine quality data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val modelTrainTest =
      (trainTest: (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )) => {

        val gr = FFNeuralGraph(
          trainTest._1._1.head._1.length,
          1,
          hidden,
          acts,
          nCounts
        )

        val transform = DataPipe(
          (d: Stream[(BDV[Double], Double)]) => d.map(el => (el._1, BDV(el._2)))
        )

        val model = new FeedForwardNetwork[
          Stream[(BDV[Double], Double)]
        ](trainTest._1._1.toStream, gr)(transform)

        model
          .setLearningRate(stepSize)
          .setMaxIterations(maxIt)
          .setBatchFraction(mini)
          .setMomentum(alpha)
          .setRegParam(regularization)
          .learn()

        val res = model.test(trainTest._1._2.toStream)

        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(BDV[Double], BDV[Double])]) =>
              res.map(i => (i._1(0), i._2(0))).toList
          )

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val flag = if (acts.last == "logsig") {
          true
        } else {
          false
        }

        val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length,
          logisticFlag = flag
        )

        metrics.setName(wineType + " wine quality")
        metrics.print()
        metrics.generatePlots()
      }

    val processLabelsinPatterns = acts.last match {
      case "tansig" =>
        IterableDataPipe(
          (pattern: (BDV[Double], Double)) =>
            if (pattern._2 <= 6.0) (pattern._1, -1.0) else (pattern._1, 1.0)
        )
      case "linear" =>
        IterableDataPipe(
          (pattern: (BDV[Double], Double)) =>
            if (pattern._2 <= 6.0) (pattern._1, -1.0) else (pattern._1, 1.0)
        )
      case "logsig" =>
        IterableDataPipe(
          (pattern: (BDV[Double], Double)) =>
            if (pattern._2 <= 6.0) (pattern._1, 0.0) else (pattern._1, 1.0)
        )
    }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.dropHead >
      DynaMLPipe.replace(";", ",") >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets >
      processLabelsinPatterns

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DynaMLPipe.splitTrainingTest(training, test) >
      DynaMLPipe.featuresGaussianStandardization >
      DataPipe(modelTrainTest)

    val dataFile = dataDir + "winequality-" + wineType + ".csv"

    trainTestPipe(dataFile, dataFile)

  }

}

object TestLogisticWineQuality {
  def apply(
    training: Int = 100,
    test: Int = 1000,
    columns: List[Int] = List(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    stepSize: Double = 0.01,
    maxIt: Int = 30,
    mini: Double = 1.0,
    regularization: Double = 0.5,
    wineType: String = "red",
    modelType: String = "logistic"
  ): Unit = {

    //Load wine quality data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val modelpipe = new GLMPipe[
      Stream[(BDV[Double], Double)],
      (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )
    ](
      (tt: (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )) => tt._1._1.toStream,
      task = "classification",
      modelType = modelType
    ) >
      DynaMLPipe.trainParametricModel[
        Stream[(BDV[Double], Double)],
        BDV[Double],
        BDV[Double],
        Double,
        Stream[(BDV[Double], Double)],
        GeneralizedLinearModel[Stream[(BDV[Double], Double)]]
      ](regularization, stepSize, maxIt, mini)

    val testPipe = DataPipe(
      (modelAndData: (
        GeneralizedLinearModel[Stream[(BDV[Double], Double)]],
        Iterable[(BDV[Double], Double)]
      )) => {

        val pipe1 = StreamDataPipe((couple: (BDV[Double], Double)) => {
          (modelAndData._1.predict(couple._1), couple._2)
        })

        val scoresAndLabelsPipe = pipe1
        val scoresAndLabels =
          scoresAndLabelsPipe.run(modelAndData._2.toStream).toList

        val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length,
          logisticFlag = true
        )

        metrics.setName(wineType + " wine quality")
        metrics.print()
        metrics.generatePlots()

      }
    )

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.dropHead >
      DynaMLPipe.replace(";", ",") >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets >
      IterableDataPipe(
        (pattern: (BDV[Double], Double)) =>
          if (pattern._2 <= 6.0) (pattern._1, 0.0) else (pattern._1, 1.0)
      )

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DynaMLPipe.splitTrainingTest(training, test) >
      DynaMLPipe.featuresGaussianStandardization >
      BifurcationPipe(
        modelpipe,
        DataPipe(
          (tt: (
            (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
            (BDV[Double], BDV[Double])
          )) => tt._1._2
        )
      ) >
      testPipe

    val dataFile = dataDir + "winequality-" + wineType + ".csv"

    trainTestPipe(dataFile, dataFile)

  }

}

object TestLSSVMWineQuality {
  def apply(
    kernel: LocalSVMKernel[BDV[Double]],
    training: Int = 100,
    test: Int = 1000,
    columns: List[Int] = List(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    regularization: Double = 0.5,
    wineType: String = "red"
  ): Unit = {

    //Load wine quality data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val modelTrainTest =
      (trainTest: (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )) => {

        val model = new DLSSVM(
          trainTest._1._1.toStream,
          training,
          kernel,
          "classification"
        )

        model.setRegParam(regularization).learn()

        val pipe1 = StreamDataPipe((couple: (BDV[Double], Double)) => {
          (model.predict(couple._1), couple._2)
        })

        val scoresAndLabelsPipe = pipe1
        val scoresAndLabels =
          scoresAndLabelsPipe.run(trainTest._1._2.toStream).toList

        val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length,
          logisticFlag = true
        )

        metrics.setName(wineType + " wine quality")
        metrics.print()
        metrics.generatePlots()
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.dropHead >
      DynaMLPipe.replace(";", ",") >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets >
      IterableDataPipe(
        (pattern: (BDV[Double], Double)) =>
          if (pattern._2 <= 6.0) (pattern._1, -1.0) else (pattern._1, 1.0)
      )

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DynaMLPipe.splitTrainingTest(training, test) >
      DynaMLPipe.featuresGaussianStandardization >
      DataPipe(modelTrainTest)

    val dataFile = dataDir + "winequality-" + wineType + ".csv"

    trainTestPipe(dataFile, dataFile)

  }
}
