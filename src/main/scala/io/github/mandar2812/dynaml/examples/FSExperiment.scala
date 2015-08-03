package io.github.mandar2812.dynaml.examples

import java.io.File

import breeze.linalg.DenseVector
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.utils
import scala.collection.mutable.{MutableList => ML}

/**
 * @author mandar2812 on 3/8/15.
 * Carry out experiment with the FS-LSSVM
 * on three possible data sets.
 * 1. Forest Cover type : Class 2 vs rest
 * 2. Adult Data set
 * 3. Magic Gamma Telescope Data set.
 *
 * Conduct a number of trials and average out
 * the performance results and store it in a file
 */
object FSExperiment {
  def apply(nCores: Int = 4, trials: Int,
            data: String = "ForestCover",
            root: String = "data/"): Unit = {
    val writer = CSVWriter.open(new File(root+data+".csv"), append = true)
    List("gs", "csa").foreach((globalOpt) => {
      List(50, 100, 200, 300, 500).foreach((prototypes) => {
        List("RBF", "Polynomial", "Laplacian", "Linear").foreach((kern) => {
          List(2, 3).foreach((gridSize) => {
            val perfs: ML[DenseVector[Double]] = ML()
            var times: ML[Double] = ML()
            (1 to trials).toList.foreach((trial) => {
              data match {
                case "ForestCover" => {
                  val t0 = System.currentTimeMillis().toDouble
                  perfs += TestForestCover(nCores,
                    prototypes, kern,
                    globalOpt, grid = gridSize,
                    frac = 1.0, dataRoot = root, local = true)/trials.toDouble
                  times += System.currentTimeMillis().toDouble - t0
                }
                case "Adult" => {
                  val t0 = System.currentTimeMillis().toDouble
                  perfs += TestAdult(nCores,
                    prototypes, kern,
                    globalOpt, grid = gridSize,
                    frac = 1.0)/trials.toDouble
                  times += System.currentTimeMillis().toDouble - t0
                }
                case "MagicGamma" => {
                  val t0 = System.currentTimeMillis().toDouble
                  perfs += TestMagicGamma(nCores,
                    prototypes, kern,
                    globalOpt, grid = gridSize,
                    dataRoot = root)/trials.toDouble
                  times += System.currentTimeMillis().toDouble - t0
                }
              }
            })
            val (avg_perf, var_perf) = utils.getStats(perfs.toList)
            val (avg_time, var_time) = utils.getStats(times.toList.map(DenseVector(_)))
            val row = Seq(kern, prototypes.toString, globalOpt,
              gridSize.toString, 0.45, "linear",
              avg_perf(0), math.sqrt(var_perf(0)/(trials.toDouble-1.0)),
              avg_perf(1), math.sqrt(var_perf(1)/(trials.toDouble-1.0)),
              avg_perf(2), math.sqrt(var_perf(2)/(trials.toDouble-1.0)),
              avg_time(0), math.sqrt(var_time(0)/(trials.toDouble-1.0)))
            writer.writeRow(row)
          })
        })
      })
    })
    writer.close()
  }
}
