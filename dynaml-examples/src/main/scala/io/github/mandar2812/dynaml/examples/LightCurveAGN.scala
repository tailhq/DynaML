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
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.log4j.Logger
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, LocalScalarKernel}
import io.github.mandar2812.dynaml.models.gp.{GPNarModel, GPTimeSeries}
import io.github.mandar2812.dynaml.optimization.{GradBasedGlobalOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.DataPipe


/**
  * Created by mandar on 4/3/16.
  */
object LightCurveAGN {

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
    //Load data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]))) => {

        val model = new GPNarModel(deltaT, kernel,
          noise, trainTest._1)

        val gs = opt("globalOpt") match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "ML" => new GradBasedGlobalOptimizer[model.type](model)
        }

        val startConf = kernel.state ++ noise.state

        val (_, conf) = gs.optimize(startConf, opt)

        val res = model.test(trainTest._2.toSeq)

        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
              res.map(i => (i._3, i._2, i._4, i._5)).toList)

        val scoresAndLabels = scoresAndLabelsPipe.run(res.toList)

        val metrics = new RegressionMetrics(scoresAndLabels.map(i => (i._1, i._2)),
          scoresAndLabels.length)

        val name = "Light Curve"
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
        title("Active Galactic Nucleus "+opt("objectId").capitalize.replace("_", " ")+": "+name)
        unhold()

        Seq(
          Seq(deltaT, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines > DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0, 1),
        Map()
      ) > DataPipe((lines: Stream[String]) => lines.map{line =>
        val splits = line.split(",")
        (splits.head.toDouble, splits.last.toDouble)
      }) > DynaMLPipe.deltaOperation(deltaT, 0)

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/"+opt("objectId")+".csv",
      "data/"+opt("objectId")+".csv"))

  }

  def runExperimentTS(kernel: LocalScalarKernel[Double],
                      noise: LocalScalarKernel[Double],
                      deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                      num_training: Int = 150, num_test:Int,
                      opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(Double, Double)],
        Stream[(Double, Double)]))) => {

        val model = new GPTimeSeries(kernel,
          noise, trainTest._1.toSeq)

        val gs = opt("globalOpt") match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "ML" => new GradBasedGlobalOptimizer[model.type](model)
        }

        val startConf = kernel.state ++ noise.state

        val (_, conf) = gs.optimize(startConf, opt)

        val res = model.test(trainTest._2.toSeq)

        val scoresAndLabelsPipe =
          DataPipe((res: Seq[(Double, Double, Double, Double, Double)]) =>
            res.map(i => (i._3, i._2, i._4, i._5)).toList)

        val scoresAndLabels = scoresAndLabelsPipe.run(res.toList)

        val metrics = new RegressionMetrics(scoresAndLabels.map(i => (i._1, i._2)),
          scoresAndLabels.length)

        val name = "Light Curve"
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
        title("Active Galactic Nucleus "+opt("objectId").capitalize.replace("_", " ")+": "+name)
        unhold()

        Seq(
          Seq(deltaT, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines > DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0, 1),
        Map()
      ) > DataPipe((lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      (splits.head.toDouble, splits.last.toDouble)
    })

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.duplicate(
      DataPipe((l: Stream[(Double, Double)]) =>
        l.map(d => (DenseVector(d._1), d._2)))
      ) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DataPipe((X: (Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)])) => {
        (X._1.map(c => (c._1(0), c._2)),
          X._2.map(c => (c._1(0), c._2)))
      }) >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/"+opt("objectId")+".csv",
      "data/"+opt("objectId")+".csv"))

  }



}
