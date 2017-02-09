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
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.GPRegressionPipe
import io.github.mandar2812.dynaml.models.gp.{AbstractGPRegressionModel, GPRegression}
import io.github.mandar2812.dynaml.pipes.{BifurcationPipe, DataPipe}

/**
  * Created by mandar on 15/12/15.
  */
object TestGPHousing {
  
  
  type Features = DenseVector[Double]
  type Output = Double
  type Pattern = (Features, Output)
  type Data = Stream[Pattern]
  type DataAlt = Seq[Pattern]
  type TTData = (Data, Data)
  type Kernel = LocalScalarKernel[Features]
  type Scales = (Features, Features)
  type GP = AbstractGPRegressionModel[Data, Features]
  type GPAlt = AbstractGPRegressionModel[DataAlt, Features]
  type PredictionsAndErrBars = Seq[(Features, Output, Output, Output, Output)]
  type PredictionsAndOutputs = List[(Output, Output)]
  

  def apply(kernel: Kernel,
            bandwidth: Double = 0.5,
            noise: Kernel,
            trainFraction: Double = 0.75,
            columns: List[Int] = List(13,0,1,2,3,4,5,6,7,8,9,10,11,12),
            grid: Int = 5, step: Double = 0.2, globalOpt: String = "ML",
            stepSize: Double = 0.01, maxIt: Int = 300, policy: String = "GS"): Unit =
    runExperiment(kernel, bandwidth,
      noise, (506*trainFraction).toInt, columns,
      grid, step, globalOpt, policy,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString
      )
    )

  def runExperiment(kernel: Kernel,
                    bandwidth: Double = 0.5,
                    noise: Kernel,
                    num_training: Int = 200, columns: List[Int] = List(40,16,21,23,24,22,25),
                    grid: Int = 5, step: Double = 0.2,
                    globalOpt: String = "ML", pol: String = "GS",
                    opt: Map[String, String]): Unit = {


    val startConf = kernel.effective_state ++ noise.effective_state

    val modelpipe =
      GPRegressionPipe[(TTData, Scales), Features]((tt: (TTData, Scales)) => tt._1._1, kernel, noise) >
      gpTuning[DataAlt, Features](startConf, globalOpt, grid, step, opt("maxIterations").toInt, pol) >
      DataPipe((modelCouple: (GPAlt, Map[String, Double])) => {
        modelCouple._1
      })

    val testPipe = DataPipe((testSample: (GPAlt, (Data, Scales))) => {
      val (model, (data, scales)) = testSample
      (model.test(data), scales) }) >
      DataPipe((res: (PredictionsAndErrBars, Scales)) =>
        res._1
          .map(i => (i._3, i._2)).toList
          .map(l => (l._1*res._2._2(-1) + res._2._1(-1), l._2*res._2._2(-1) + res._2._1(-1)))) >
      DataPipe((scoresAndLabels: PredictionsAndOutputs) => {

        val metrics = new RegressionMetrics(scoresAndLabels, scoresAndLabels.length)
        metrics.setName("MEDV")
        //Print the evaluation results
        metrics.print()
        metrics.generatePlots()
      })


    //Load Housing data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preProcessPipe = fileToStream >
      trimLines >
      replaceWhiteSpaces >
      extractTrainingFeatures(columns, Map()) >
      splitFeaturesAndTargets

    val trainTestPipe = duplicate(preProcessPipe) >
      splitTrainingTest(num_training, 506-num_training) >
      trainTestGaussianStandardization >
      BifurcationPipe(
        modelpipe,
        DataPipe((tt: (TTData, Scales)) => (tt._1._2, tt._2))
      ) >
      testPipe

    trainTestPipe.run(("data/housing.data", "data/housing.data"))

  }

}
