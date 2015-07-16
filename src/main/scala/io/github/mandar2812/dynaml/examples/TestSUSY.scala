package io.github.mandar2812.dynaml.examples

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.kernels.{SVMKernel, RBFKernel}
import io.github.mandar2812.dynaml.models.KernelizedModel
import io.github.mandar2812.dynaml.models.svm.{LSSVMSparkModel, KernelSparkModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * @author mandar2812 on 1/7/15.
 */
object TestSUSY {
  def main(args: Array[String]) = {

    val prot = args(0).toInt
    val kern = args(1)
    val go = args(2)
    val grid = args(3).toInt
    val step = args(4).toDouble
    val dataRoot = args(5)
    TestSUSY(4, prot, kern, go,
      grid, step, true, 1.0, dataRoot)
  }

  def apply(nCores: Int = 4, prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.3, logscale: Boolean = false, frac: Double, root: String): Unit = {

    val trainFile = root+"susy.csv"
    val testFile = root+"susytest.csv"
    val config = Map(
      "file" -> trainFile,
      "delim" -> ",",
      "head" -> "false",
      "task" -> "classification",
      "parallelism" -> nCores.toString
    )

    val configtest = Map("file" -> testFile,
      "delim" -> ",",
      "head" -> "false")

    val conf = new SparkConf().setAppName("SUSY")

    conf.registerKryoClasses(Array(classOf[LSSVMSparkModel], classOf[KernelSparkModel],
      classOf[KernelizedModel[RDD[(Long, LabeledPoint)], RDD[LabeledPoint],
        DenseVector[Double], DenseVector[Double], Double, Int, Int]],
      classOf[SVMKernel[DenseMatrix[Double]]], classOf[RBFKernel],
      classOf[DenseVector[Double]],
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

    val writer = CSVWriter.open(new File(root+"resultsSUSY.csv"), append = true)
    writer.writeRow(row)
    writer.close()
    optModel.unpersist

  }
}
