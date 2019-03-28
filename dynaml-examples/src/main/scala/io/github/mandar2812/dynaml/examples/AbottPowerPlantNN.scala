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
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import io.github.mandar2812.dynaml.pipes._
import org.apache.log4j.Logger
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork

import scala.collection.mutable.{MutableList => ML}

/**
  * Created by mandar on 15/12/15.
  */
object AbottPowerPlantNN {

  private val logger = Logger.getLogger(this.getClass)

  implicit val transform = DataPipe(
    (d: Stream[(DenseVector[Double], DenseVector[Double])]) => d
  )

  def apply(
    delta: Int,
    hidden: Int = 2,
    nCounts: List[Int] = List(),
    acts: List[String],
    num_training: Int = 2000,
    num_test: Int = 1000,
    stepSize: Double = 0.01,
    maxIt: Int = 300,
    mini: Double = 1.0,
    alpha: Double = 0.0,
    regularization: Double = 0.5
  ): Unit =
    runExperiment(
      hidden,
      nCounts,
      acts,
      num_training,
      num_test,
      delta,
      Map(
        "tolerance"         -> "0.0001",
        "step"              -> stepSize.toString,
        "maxIterations"     -> maxIt.toString,
        "miniBatchFraction" -> mini.toString,
        "momentum"          -> alpha.toString,
        "regularization"    -> regularization.toString
      )
    )

  def runExperiment(
    hidden: Int = 2,
    nCounts: List[Int] = List(),
    act: List[String],
    num_training: Int = 200,
    num_test: Int,
    deltaT: Int = 2,
    opt: Map[String, String]
  ): Unit = {

    val names = Map(
      5 -> "Drum pressure PSI",
      6 -> "Excess Oxygen",
      7 -> "Water level in Drum",
      8 -> "Steam Flow kg/s"
    )

    val modelTrainTest =
      (trainTest: (
        (
          Iterable[(DenseVector[Double], DenseVector[Double])],
          Iterable[(DenseVector[Double], DenseVector[Double])]
        ),
        (DenseVector[Double], DenseVector[Double])
      )) => {

        logger.info("Number of Inputs: " + trainTest._1._1.head._1.length)
        logger.info("Number of Outputs: " + trainTest._1._1.head._2.length)

        val gr =
          FFNeuralGraph(trainTest._1._1.head._1.length, 4, hidden, act, nCounts)

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], DenseVector[Double])]
        ](trainTest._1._1.toStream, gr)(transform)

        model
          .setLearningRate(opt("step").toDouble)
          .setMaxIterations(opt("maxIterations").toInt)
          .setBatchFraction(opt("miniBatchFraction").toDouble)
          .setMomentum(opt("momentum").toDouble)
          .setRegParam(opt("regularization").toDouble)
          .learn()

        val res = model.test(trainTest._1._2.toStream)

        val l = trainTest._1._1.head._1.length +
          trainTest._1._1.head._2.length

        val means   = trainTest._2._1((l - 4) until l)
        val stdDevs = trainTest._2._2((l - 4) until l)

        val scoresAndLabelsPipe1 = DataPipe(
          (res: Seq[(DenseVector[Double], DenseVector[Double])]) => {
            res.map(r => ((r._1 :* stdDevs) + means, (r._2 :* stdDevs) + means))
          }
        ) >
          DataPipe((res: Seq[(DenseVector[Double], DenseVector[Double])]) => {
            val num_outputs = res.head._1.length
            val outputAcc: List[ML[(Double, Double)]] =
              List(ML(), ML(), ML(), ML())
            res.foreach(r => {
              (0 until num_outputs).foreach(output => {
                outputAcc(output) ++= ML((r._1(output), r._2(output)))
              })
            })
            outputAcc.map(_.toList)
          })

        val scoresAndLabels = scoresAndLabelsPipe1.run(res)

        var index = 5

        scoresAndLabels.foreach((output) => {
          val metrics = new RegressionMetrics(output, output.length)
          metrics.setName(names(index))
          metrics.print()
          metrics.generateFitPlot()
          //Plotting time series prediction comparisons
          line((1 to output.length).toList, output.map(_._2))
          hold()
          line((1 to output.length).toList, output.map(_._1))
          legend(
            List(
              names(index),
              "Predicted " + names(index) + " (one hour ahead)"
            )
          )
          title(
            "Steam Generator; Abbott Power Plant, Champaign IL: " + names(index)
          )
          unhold()
          index += 1
        })

      }

    //Load Abott power plant data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0, 5, 6, 7, 8, 1, 2, 3, 4),
        Map()
      ) >
      DynaMLPipe.removeMissingLines >
      IterableDataPipe((line: String) => {
        val splits    = line.split(",")
        val timestamp = splits.head.toDouble
        val feat      = DenseVector(splits.tail.map(_.toDouble))
        (timestamp, feat)
      }) >
      DataPipe(
        (lines: Iterable[(Double, DenseVector[Double])]) =>
          lines.toList
            .sliding(deltaT + 1)
            .map((history) => {
              val hist                    = history.take(history.length - 1).map(_._2)
              val featuresAcc: ML[Double] = ML()

              (0 until hist.head.length).foreach((dimension) => {
                //for each dimension/regressor take points t to t-order
                featuresAcc ++= hist.map(vec => vec(dimension))
              })

              val features = DenseVector(featuresAcc.toArray)
              //assert(history.length == deltaT + 1, "Check one")
              //assert(features.length == deltaT, "Check two")
              (features, history.last._2(0 to 3))
            })
            .toStream
      )

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DataPipe(
        (data: (
          Stream[(DenseVector[Double], DenseVector[Double])],
          Stream[(DenseVector[Double], DenseVector[Double])]
        )) => {
          (data._1.take(num_training), data._2.takeRight(num_test))
        }
      ) >
      DynaMLPipe.trainTestGaussianStandardizationMO >
      DataPipe(modelTrainTest)

    val dataFile = dataDir + "/steamgen.csv"
    trainTestPipe((dataFile, dataFile))

  }

}
