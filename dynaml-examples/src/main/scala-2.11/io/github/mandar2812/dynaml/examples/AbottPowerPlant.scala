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
import io.github.mandar2812.dynaml.evaluation.{MultiRegressionMetrics, RegressionMetrics}
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.gp.{AbstractGPRegressionModel, GPNarXModel, GPRegression}
import io.github.mandar2812.dynaml.models.stp.MVStudentsTModel
import io.github.mandar2812.dynaml.optimization.{GlobalOptimizer, GradBasedGlobalOptimizer, GridSearch, ProbGPCommMachine}
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler

import scala.collection.mutable.{MutableList => ML}

/**
  * @author mandar2812 date 4/3/16.
  * */
object AbottPowerPlant {

  type Features = DenseVector[Double]
  type Kernel = LocalScalarKernel[Features]
  type Scales = (GaussianScaler, GaussianScaler)
  type GPModel = AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)], DenseVector[Double]]

  type Data = Stream[(Features, Features)]

  val names = Map(
    5 -> "Drum pressure PSI",
    6 -> "Excess Oxygen",
    7 -> "Water level in Drum",
    8 -> "Steam Flow kg/s")

  val deltaOperationARXMultiOutput = (deltaT: List[Int], deltaTargets: List[Int]) =>
    DataPipe((lines: Stream[(Double, DenseVector[Double])]) =>
      lines.toList.sliding(math.max(deltaT.max, deltaTargets.max) + 1).map((history) => {

        val hist = history.take(history.length - 1).map(_._2)

        val num_outputs = deltaTargets.length

        val featuresAcc: ML[Double] = ML()

        val lags = deltaT ++ deltaTargets

        (0 until hist.head.length).foreach((dimension) => {
          //for each dimension/regressor take points t to t-order
          featuresAcc ++= hist.takeRight(lags(dimension))
            .map(vec => vec(dimension))
        })

        val outputs = history.last._2(0 until num_outputs)
        val features = DenseVector(featuresAcc.toArray)

        (features, outputs)
      }).toStream)


  def apply(
    kernel: Kernel, noise: Kernel,
    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
    num_training: Int = 150, num_test:Int = 1000, column: Int = 7,
    opt: Map[String, String]) = runExperiment(
    kernel, noise, deltaT, timelag,
    stepPred, num_training, num_test, column, opt)



  def apply(
    kernel: Kernel, noise: Kernel)(
    deltaT: Int, num_training: Int, num_test: Int,
    opt: Map[String, String]) =
    runMOExperiment(kernel, noise, deltaT, num_training, num_test, opt)

  def runMOExperiment(
    kernel: Kernel, noise: Kernel, deltaT: Int = 2,
    num_training: Int = 150, num_test: Int,
    opt: Map[String, String]) = {


    val preProcessPipe = fileToStream >
      trimLines >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        List(0,5,6,7,8,1,2,3,4),
        Map()
      ) >
      removeMissingLines >
      StreamDataPipe((line: String) => {
        val splits = line.split(",")
        val timestamp = splits.head.toDouble
        val data_vec = DenseVector(splits.tail.map(_.toDouble))
        (timestamp, data_vec)
      }) >
      deltaOperationARXMultiOutput(
        List.fill(4)(deltaT),
        List.fill(4)(deltaT))


    val modelTrainTest = DataPipe((dataAndScales: (Data, Data, Scales)) => {

      val (training_data, test_data, scales) = dataAndScales

      val targetScales = scales._2

      val reScaleTargets = targetScales.i * targetScales.i

      implicit val transform = DataPipe((d: Data) => d.toSeq)

      val multioutputTModel = MVStudentsTModel[Data, Features](kernel, noise, identityPipe[Features]) _

      val model = multioutputTModel(training_data, training_data.length, training_data.head._2.length)

      val gs = opt("globalOpt") match {
        case "GS" => new GridSearch[model.type](model)
          .setGridSize(opt("grid").toInt)
          .setStepSize(opt("step").toDouble)
          .setLogScale(false)

      }

      val startConf = kernel.effective_state ++ noise.effective_state

      val (optModel, _) = gs.optimize(startConf, opt)

      val res = optModel.test(test_data).map(t => (t._3, t._2)).toList

      val metrics = new MultiRegressionMetrics(reScaleTargets(res), res.length)

      (optModel, metrics)
    })


    val trainTestPipe = duplicate(preProcessPipe) >
      splitTrainingTest(num_training, num_test) >
      gaussianScalingTrainTest >
      modelTrainTest

    trainTestPipe.run(
      ("data/steamgen.csv",
        "data/steamgen.csv"))

  }

  def runExperiment(
    kernel: Kernel, noise: Kernel,
    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
    num_training: Int = 150, num_test:Int, column: Int = 7,
    opt: Map[String, String]): Seq[Seq[AnyVal]] = {


    //Load Abott power plant data into a stream
    //Extract the time and target values
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val modelTrainTest =
      (trainTest: ((Stream[(Features, Double)], Stream[(Features, Double)]), (Features, Features))) => {

        val model = new GPNarXModel(deltaT, 4, kernel,
          noise, trainTest._1._1)

        val gs: GlobalOptimizer[GPModel] = opt("globalOpt") match {
          case "GS" => new GridSearch[GPModel](model)
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)

          case "ML" => new GradBasedGlobalOptimizer(model)

          case "GPC" => new ProbGPCommMachine(model)
            .setPolicy(opt("policy"))
            .setGridSize(opt("grid").toInt)
            .setStepSize(opt("step").toDouble)
            .setLogScale(false)
            .setMaxIterations(opt("maxIterations").toInt)

        }

        val startConf = kernel.effective_state ++ noise.effective_state

        val (optModel, _): (GPModel, Map[String, Double]) = gs.optimize(startConf, opt)

        val res = optModel.test(trainTest._1._2)

        val deNormalize = DataPipe((list: List[(Double, Double, Double, Double)]) =>
          list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._2*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._3*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._4*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabelsPipe =
          DataPipe[
            Seq[(Features, Double, Double, Double, Double)],
            List[(Double, Double, Double, Double)]](
            _.map(i => (i._3, i._2, i._4, i._5)).toList) >
            deNormalize

        val scoresAndLabels = scoresAndLabelsPipe.run(res.toList)

        val metrics = new RegressionMetrics(
          scoresAndLabels.map(i => (i._1, i._2)),
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

    val dataFile = dataDir+"/steamgen.csv"
    trainTestPipe((dataFile, dataFile))
  }
}
