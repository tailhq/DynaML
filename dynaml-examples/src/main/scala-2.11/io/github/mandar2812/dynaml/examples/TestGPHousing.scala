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
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.GPRegressionPipe
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{BifurcationPipe, DataPipe}

/**
  * Created by mandar on 15/12/15.
  */
object TestGPHousing {

  def apply(kernel: LocalScalarKernel[DenseVector[Double]],
            bandwidth: Double = 0.5,
            noise: LocalScalarKernel[DenseVector[Double]],
            trainFraction: Double = 0.75,
            columns: List[Int] = List(13,0,1,2,3,4,5,6,7,8,9,10,11,12),
            grid: Int = 5, step: Double = 0.2, globalOpt: String = "ML",
            stepSize: Double = 0.01, maxIt: Int = 300): Unit =
    runExperiment(kernel, bandwidth,
      noise, (506*trainFraction).toInt, columns,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString
      )
    )

  def apply(kern: String,
            bandwidth: Double,
            noise: LocalScalarKernel[DenseVector[Double]],
            trainFraction: Double,
            columns: List[Int],
            grid: Int, step: Double, globalOpt: String,
            stepSize: Double, maxIt: Int): Unit = {

    implicit val field = VectorField(columns.length - 1)

    val kernel: LocalScalarKernel[DenseVector[Double]] =
      kern match {
        case "RBF" =>
          new RBFKernel(bandwidth)
        case "Cauchy" =>
          new CauchyKernel(bandwidth)
        case "Laplacian" =>
          new LaplacianKernel(bandwidth)
        case "RationalQuadratic" =>
          new RationalQuadraticKernel(bandwidth)
        case "FBM" => new FBMKernel(bandwidth)
        case "Student" => new TStudentKernel(bandwidth)
        case "Periodic" => new PeriodicKernel(bandwidth, bandwidth)
      }

    val num_training = 506*trainFraction

    runExperiment(kernel, bandwidth,
      noise, num_training.toInt, columns,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString
      )
    )

  }

  def runExperiment(kernel: LocalScalarKernel[DenseVector[Double]],
                    bandwidth: Double = 0.5,
                    noise: LocalScalarKernel[DenseVector[Double]],
                    num_training: Int = 200, columns: List[Int] = List(40,16,21,23,24,22,25),
                    grid: Int = 5, step: Double = 0.2,
                    globalOpt: String = "ML", opt: Map[String, String]): Unit = {


    val startConf = kernel.state ++ noise.state

    val modelpipe = new GPRegressionPipe[
      GPRegression, ((Stream[(DenseVector[Double], Double)],
      Stream[(DenseVector[Double], Double)]),
      (DenseVector[Double], DenseVector[Double]))](
      (tt: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => tt._1._1,
      kernel, noise) >
      DynaMLPipe.modelTuning(startConf, globalOpt, grid, step) >
      DataPipe((modelCouple: (GPRegression, Map[String, Double])) => {
        modelCouple._1.setState(modelCouple._2)
        modelCouple._1
      })

    val testPipe = DataPipe((testSample: (GPRegression,
          (Stream[(DenseVector[Double], Double)],
          (DenseVector[Double], DenseVector[Double]))
          )) => (testSample._1.test(testSample._2._1), testSample._2._2)) >
        DataPipe((res: (Seq[(DenseVector[Double], Double, Double, Double, Double)],
          (DenseVector[Double], DenseVector[Double]))) =>
          res._1.map(i => (i._3, i._2)).toList.map{l => (l._1*res._2._2(-1) + res._2._1(-1),
            l._2*res._2._2(-1) + res._2._1(-1))}
        ) > DataPipe((scoresAndLabels: List[(Double, Double)]) => {
        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)

        metrics.setName("MEDV")

        metrics.print()
        metrics.generatePlots()
      })


    //Load Housing data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, 506-num_training) >
      DynaMLPipe.trainTestGaussianStandardization >
      BifurcationPipe(
        modelpipe,
        DataPipe((tt: (
          (Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)]),
            (DenseVector[Double], DenseVector[Double]))) => (tt._1._2, tt._2)
        )
      ) > testPipe

    trainTestPipe.run(("data/housing.data", "data/housing.data"))

  }

}
