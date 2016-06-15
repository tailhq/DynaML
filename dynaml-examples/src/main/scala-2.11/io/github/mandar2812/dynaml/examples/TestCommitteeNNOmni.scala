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
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.models.neuralnets.{CommitteeNetwork, FFNeuralGraph}
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.log4j.Logger

/**
  * Created by mandar on 11/2/16.
  */
object TestCommitteeNNOmni {

  def apply(year: Int, yeartest:Int,
            hidden: Int = 1, nCounts:List[Int] = List(), acts:List[String],
            delta: Int, timeLag:Int, stepAhead: Int,
            num_training: Int, num_test: Int,
            column: Int, stepSize: Double = 0.05,
            maxIt: Int = 200, mini: Double = 1.0,
            alpha: Double = 0.0, regularization: Double = 0.5): Unit =
    runExperiment(year, yeartest, hidden, nCounts, acts,
      delta, timeLag, stepAhead,
      num_training, num_test, column,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString,
        "miniBatchFraction" -> mini.toString,
        "momentum" -> alpha.toString,
        "regularization" -> regularization.toString
      ))

  def runExperiment(year: Int = 2006, yearTest:Int = 2007,
                    hidden: Int = 2, nCounts:List[Int] = List(), act:List[String],
                    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                    num_training: Int = 200, num_test: Int = 50,
                    column: Int = 40, opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Omni data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {


        val configs = for (c <- nCounts; a <- act) yield(c,a)

        val networks = configs.map(couple => {
          FFNeuralGraph(trainTest._1._1.head._1.length, 1, 1,
            List(couple._2, "recLinear"),List(couple._1))
        })

        val transform = DataPipe((d: Stream[(DenseVector[Double], Double)]) =>
          d.map(el => (el._1, DenseVector(el._2))))

        val model =
          new CommitteeNetwork[Stream[(DenseVector[Double], Double)]](trainTest._1._1, transform, networks:_*)

        model.baseOptimizer.setStepSize(opt("step").toDouble)
          .setNumIterations(opt("maxIterations").toInt)
          .setMomentum(opt("momentum").toDouble)
          .setRegParam(opt("regularization").toDouble)

        model.learn()

        val res = model.test(trainTest._1._2)
        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], DenseVector[Double])]) =>
              res.map(i => (i._1(0), i._2(0))).toList) >
            DataPipe((list: List[(Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)

        metrics.print()
        metrics.generatePlots()

        //Plotting time series prediction comparisons
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        legend(List("Time Series", "Predicted Time Series (one hour ahead)"))
        unhold()

        val incrementsPipe = DataPipe((list: List[(Double, Double)]) =>
              list.sliding(2).map(i => (i(1)._1 - i.head._1,
                i(1)._2 - i.head._2)).toList)

        val increments = incrementsPipe.run(scoresAndLabels)

        val incrementMetrics = new RegressionMetrics(increments, increments.length)

        logger.info("Results for Prediction of increments")
        incrementMetrics.print()
        incrementMetrics.generatePlots()

        line((1 to increments.length).toList, increments.map(_._2))
        hold()
        line((1 to increments.length).toList, increments.map(_._1))
        legend(List("Increment Time Series", "Predicted Increment Time Series (one hour ahead)"))
        unhold()

        Seq(
          Seq(year, yearTest, deltaT, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0,1,2,column),
        Map(
          16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")
      ) > DynaMLPipe.removeMissingLines >
      DynaMLPipe.extractTimeSeries((year,day,hour) => (day * 24) + hour) >
      DynaMLPipe.deltaOperation(deltaT, timelag)

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/omni2_"+year+".csv",
      "data/omni2_"+yearTest+".csv"))

  }
}

