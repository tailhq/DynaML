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

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import io.github.mandar2812.dynaml.kernels.{RBFKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.{GLMPipe, KernelizedModel}
import io.github.mandar2812.dynaml.models.lm.GeneralizedLinearModel
import io.github.mandar2812.dynaml.models.svm.{KernelSparkModel, LSSVMSparkModel}
import io.github.mandar2812.dynaml.pipes._

/**
 * @author mandar2812 on 1/7/15.
 */
object TestAdult {
  def apply(nCores: Int = 4, prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.45, logscale: Boolean = false,
            frac: Double, executors: Int = 1,
            paraFactor: Int = 2): BDV[Double] = {

    val dataRoot = "data/"
    val trainfile = dataRoot+"adult.csv"
    val testfile = dataRoot+"adulttest.csv"

    val config = Map(
      "file" -> trainfile,
      "delim" -> ",",
      "head" -> "false",
      "task" -> "classification",
      "parallelism" -> nCores.toString,
      "executors" -> executors.toString,
      "factor" -> paraFactor.toString
    )

    val configtest = Map("file" -> testfile,
      "delim" -> ",",
      "head" -> "false")

    val conf = new SparkConf().setAppName("Adult").setMaster("local["+nCores+"]")

    conf.registerKryoClasses(Array(classOf[LSSVMSparkModel], classOf[KernelSparkModel],
      classOf[KernelizedModel[RDD[(Long, LabeledPoint)], RDD[LabeledPoint],
        BDV[Double], BDV[Double], Double, Int, Int]],
      classOf[SVMKernel[DenseMatrix[Double]]], classOf[RBFKernel],
      classOf[BDV[Double]],
      classOf[DenseMatrix[Double]]))

    val sc = new SparkContext(conf)

    val model = LSSVMSparkModel(config, sc)

    val nProt = if (kernel == "Linear") {
      model.npoints.toInt
    } else {
      if(prototypes > 0)
        prototypes
      else
        math.sqrt(model.npoints.toDouble).toInt
    }

    model.setBatchFraction(frac)
    val (optModel, optConfig) = KernelizedModel.getOptimizedModel[RDD[(Long, LabeledPoint)],
      RDD[LabeledPoint], model.type](model, globalOptMethod,
        kernel, nProt, grid, step, logscale)

    optModel.setMaxIterations(2).learn()

    val met = optModel.evaluate(configtest)



    met.print()
    println("Optimal Configuration: "+optConfig)
    val scale = if(logscale) "log" else "linear"

    val perf = met.kpi()
    val row = Seq(kernel, prototypes.toString, globalOptMethod,
      grid.toString, step.toString, scale,
      perf(0), perf(1), perf(2), optConfig.toString)

    val writer = CSVWriter.open(new File("data/resultsAdult.csv"), append = true)
    writer.writeRow(row)
    writer.close()
    optModel.unpersist
    perf
  }
}


object TestAdultLogistic {

  def apply(training: Int = 1000, columns: List[Int] = List(6, 0, 1, 2, 3, 4, 5),
            stepSize: Double = 0.01, maxIt: Int = 30, mini: Double = 1.0,
            regularization: Double = 0.5,
            modelType: String = "logistic") = {

    val modelpipe = new GLMPipe(
      (tt: ((Stream[(BDV[Double], Double)], Stream[(BDV[Double], Double)]),
      (BDV[Double], BDV[Double]))) => tt._1._1,
      task = "classification", modelType = modelType
    ) > DynaMLPipe.trainParametricModel[
      Stream[(BDV[Double], Double)],
      BDV[Double], BDV[Double], Double,
      Stream[(BDV[Double], Double)],
      GeneralizedLinearModel[Stream[(BDV[Double], Double)]]
      ](regularization, stepSize, maxIt, mini)

    val testPipe =  DataPipe(
      (modelAndData: (GeneralizedLinearModel[Stream[(BDV[Double], Double)]],
        Stream[(BDV[Double], Double)])) => {

        val pipe1 = StreamDataPipe((couple: (BDV[Double], Double)) => {
          (modelAndData._1.predict(couple._1), couple._2)
        })

        val scoresAndLabelsPipe = pipe1
        val scoresAndLabels = scoresAndLabelsPipe.run(modelAndData._2).toList

        val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length,
          logisticFlag = true)

        metrics.setName("Adult Income")
        metrics.print()
        metrics.generatePlots()

      })


    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets

    val scaleFeatures = StreamDataPipe((pattern:(BDV[Double], Double)) =>
      (pattern._1, math.max(pattern._2, 0.0)))


    val procTraining = preProcessPipe >
      DataPipe((data: Stream[(BDV[Double], Double)]) => data.take(training)) >
      scaleFeatures

    val procTest = preProcessPipe > scaleFeatures

    val trainTestPipe = DataPipe(procTraining, procTest) >
      DynaMLPipe.featuresGaussianStandardization >
      BifurcationPipe(modelpipe,
        DataPipe((tt: (
          (Stream[(BDV[Double], Double)], Stream[(BDV[Double], Double)]),
            (BDV[Double], BDV[Double]))) => tt._1._2)) >
      testPipe

    trainTestPipe(("data/adult.csv",
      "data/adulttest.csv"))

  }

}