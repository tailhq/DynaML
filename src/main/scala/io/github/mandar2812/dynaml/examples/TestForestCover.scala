package io.github.mandar2812.dynaml.examples

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import io.github.mandar2812.dynaml.kernels.{RBFKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.KernelizedModel
import io.github.mandar2812.dynaml.models.svm.{KernelSparkModel, LSSVMSparkModel}

/**
 * @author mandar2812 on 1/7/15.
 */
object TestForestCover {

  def main(args: Array[String]) = {

    val prot = args(0).toInt
    val kern = args(1)
    val go = args(2)
    val grid = args(3).toInt
    val step = args(4).toDouble
    val root = args(5)
    val ex = args(6).toInt
    val cores = args(7).toInt
    TestForestCover(cores, prot, kern, go,
      grid, step, false, 1.0,
      dataRoot = root, executors = ex, false)
  }

  def apply(nCores: Int = 4, prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.3, logscale: Boolean = false, frac: Double,
            dataRoot: String = "data/", executors: Int = 1,
            local: Boolean = false): Unit = {

    val trainfile = dataRoot+"cover.csv"
    val testfile = dataRoot+"covertest.csv"

    val config = Map(
      "file" -> trainfile,
      "delim" -> ",",
      "head" -> "false",
      "task" -> "classification",
      "parallelism" -> nCores.toString,
      "executors" -> executors.toString
    )

    val configtest = Map("file" -> testfile,
      "delim" -> ",",
      "head" -> "false")

    val conf = new SparkConf().setAppName("Forest Cover")

    if(local) {
      conf.setMaster("local["+nCores.toString+"]")
    }

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

    val writer = CSVWriter.open(new File("resultsForestCover.csv"), append = true)
    writer.writeRow(row)
    writer.close()
    optModel.unpersist

  }
}
