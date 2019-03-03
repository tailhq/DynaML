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
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.log4j.Logger
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, LocalScalarKernel}
import io.github.mandar2812.dynaml.models.gp.GPNarModel
import io.github.mandar2812.dynaml.optimization.{GradBasedGlobalOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.DataPipe


/**
  * Created by mandar on 4/3/16.
  */
object SantaFeLaser {
  def apply(kernel: LocalScalarKernel[DenseVector[Double]],
            noise: LocalScalarKernel[DenseVector[Double]],
            deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
            num_training: Int = 150, num_test:Int = 1000,
            opt: Map[String, String]) =
    runExperiment(kernel, noise, deltaT, timelag,
      stepPred, num_training, num_test, opt)

  def runExperiment(kernel: LocalScalarKernel[DenseVector[Double]],
                    noise: LocalScalarKernel[DenseVector[Double]],
                    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                    num_training: Int = 150, num_test:Int,
                    opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Daisy data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val model = new GPNarModel(deltaT, kernel,
          noise, trainTest._1._1)

        val gs = opt("globalOpt") match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "ML" => new GradBasedGlobalOptimizer[model.type](model)
        }

        val startConf = kernel.state ++ noise.state

        val (_, conf) = gs.optimize(startConf, opt)

        val res = model.test(trainTest._1._2.toSeq)

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

        val name = "Laser Intensity"
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
        legend(List(name, "Predicted "+name+" (one hour ahead)", "Lower Bar", "Higher Bar"))
        title("Santa Fe Infrared Laser: "+name)
        unhold()

        Seq(
          Seq(deltaT, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines >
      DynaMLPipe.extractTrainingFeatures(
        List(0),
        Map()
      ) >
      DataPipe((lines: Stream[String]) =>
        lines.zipWithIndex.map(couple =>
          (couple._2.toDouble, couple._1.toDouble))
      ) > DynaMLPipe.deltaOperation(deltaT, 0)

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/santafelaser.csv",
      "data/santafelaser.csv"))

  }
}
