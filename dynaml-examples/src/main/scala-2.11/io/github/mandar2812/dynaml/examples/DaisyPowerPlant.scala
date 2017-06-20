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

import breeze.linalg.{DenseMatrix, DenseVector}
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, LocalScalarKernel}
import io.github.mandar2812.dynaml.models.svm.DLSSVM
import io.github.mandar2812.dynaml.optimization.{GradBasedGlobalOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import org.apache.log4j.Logger

/**
  * Created by mandar on 4/3/16.
  */
object DaisyPowerPlant {
  def apply(kernel: LocalScalarKernel[DenseVector[Double]],
            deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
            num_training: Int = 150, column: Int = 7,
            opt: Map[String, String]) =
    runExperiment(kernel, deltaT, timelag,
    stepPred, num_training, column, opt)

  def runExperiment(kernel: LocalScalarKernel[DenseVector[Double]],
                    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                    num_training: Int = 150, column: Int = 7,
                    opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Daisy data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    val names = Map(6 -> "steam pressure",
      7 -> "main stem temperature",
      8 -> "reheat steam temperature")

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val model = new DLSSVM(trainTest._1._1, num_training, kernel)

        val gs = opt("globalOpt") match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "ML" => new GradBasedGlobalOptimizer[DLSSVM](model)
        }

        val startConf = kernel.state ++ Map("regularization" ->
          opt("regularization").toDouble)

        val (_, conf) = gs.optimize(startConf, opt)



        model.setRegParam(opt("regularization").toDouble).learn()

        val res = trainTest._1._2.map(testpoint => (model.predict(testpoint._1), testpoint._2))

        val scoresAndLabelsPipe = DataPipe((list: List[(Double, Double)]) =>
          list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res.toList)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)
        metrics.setName(names(column))

        metrics.print()
        metrics.generatePlots()

        //Plotting time series prediction comparisons
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        legend(List(names(column), "Predicted "+names(column)+" (one hour ahead)"))
        title("Pont-sur-Sambre 120 MW power plant: "+names(column))
        unhold()

        Seq(
          Seq(deltaT, 1, num_training, 200-num_training,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
    DynaMLPipe.trimLines >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0,column,1,2,3,4,5),
        Map()
      ) >
      DynaMLPipe.removeMissingLines >
      StreamDataPipe((line: String) => {
        val splits = line.split(",")
        val timestamp = splits.head.toDouble
        val feat = DenseVector(splits.tail.map(_.toDouble))
        (timestamp, feat)
      }) >
      DynaMLPipe.deltaOperationVec(deltaT)

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, 200-num_training) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)

    val dataFile = dataDir+"/powerplant.csv"
    trainTestPipe((dataFile, dataFile))


  }
}
