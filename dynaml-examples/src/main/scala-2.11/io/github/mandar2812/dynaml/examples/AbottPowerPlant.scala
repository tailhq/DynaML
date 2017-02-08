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

import breeze.linalg.DenseVector
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.gp.GPNarXModel
import io.github.mandar2812.dynaml.optimization.{CSAGPCommittee, GradBasedGlobalOptimizer, GridGPCommittee, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import org.apache.log4j.Logger

/**
  * Created by mandar on 4/3/16.
  */
object AbottPowerPlant {
  def apply(kernel: LocalScalarKernel[DenseVector[Double]],
            noise: LocalScalarKernel[DenseVector[Double]],
            deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
            num_training: Int = 150, num_test:Int = 1000, column: Int = 7,
            opt: Map[String, String]) =
    runExperiment(kernel, noise, deltaT, timelag,
      stepPred, num_training, num_test, column, opt)

  def runExperiment(kernel: LocalScalarKernel[DenseVector[Double]],
                    noise: LocalScalarKernel[DenseVector[Double]],
                    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                    num_training: Int = 150, num_test:Int, column: Int = 7,
                    opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Abott power plant data into a stream
    //Extract the time and target values

    val logger = Logger.getLogger(this.getClass)

    val names = Map(
      5 -> "Drum pressure PSI",
      6 -> "Excess Oxygen",
      7 -> "Water level in Drum",
      8 -> "Steam Flow kg/s")

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val model = new GPNarXModel(deltaT, 4, kernel,
          noise, trainTest._1._1)

        val gs = opt("globalOpt") match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "ML" => new GradBasedGlobalOptimizer[model.type](model)

          case "GPC" => new GridGPCommittee(model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "GPCSA" => new CSAGPCommittee(model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)
            .setMaxIterations(opt("maxIterations").toInt)

        }

        val startConf = kernel.effective_state ++ noise.effective_state

        val (optModel, conf) = gs.optimize(startConf, opt)

        val res = optModel.test(trainTest._1._2)

        val deNormalize = DataPipe((list: List[(Double, Double, Double, Double)]) =>
          list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._2*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._3*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._4*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
              res.map(i => (i._3, i._2, i._4, i._5)).toList) > deNormalize

        val scoresAndLabels = scoresAndLabelsPipe.run(res.toList)

        val metrics = new RegressionMetrics(scoresAndLabels.map(i => (i._1, i._2)),
          scoresAndLabels.length)

        val (name, name1) =
          if(names.contains(column)) (names(column), names(column))
          else ("Value","Time Series")

        metrics.setName(name)

        metrics.print()
        metrics.generateFitPlot()

        //Plotting time series prediction comparisons
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        spline((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._3))
        hold()
        spline((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._4))
        legend(List(name1, "Predicted "+name1+" (one hour ahead)", "Lower Bar", "Higher Bar"))
        title("Abott power plant, Illinois USA: "+names(column))
        unhold()

        Seq(
          Seq(deltaT, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = fileToStream >
      trimLines >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        List(0,column,1,2,3,4),
        Map()
      ) >
      removeMissingLines >
      StreamDataPipe((line: String) => {
        val splits = line.split(",")
        val timestamp = splits.head.toDouble
        val feat = DenseVector(splits.tail.map(_.toDouble))
        (timestamp, feat)
      }) >
      deltaOperationVec(deltaT)

    val trainTestPipe = duplicate(preProcessPipe) >
      splitTrainingTest(num_training, num_test) >
      trainTestGaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/steamgen.csv",
      "data/steamgen.csv"))

  }
}
