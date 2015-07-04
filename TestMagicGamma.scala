import java.io.File

import com.github.tototoshi.csv.CSVWriter
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.kuleuven.esat.models.KernelizedModel
import org.kuleuven.esat.svm.LSSVMSparkModel

/**
 * @author mandar2812
 */

object TestMagicGamma {
  def apply(nCores: Int = 4, prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.3, logscale: Boolean = false): Unit = {
    val config = Map("file" -> "data/magicgamma.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "classification",
      "parallelism" -> nCores.toString)

    val configtest = Map("file" -> "data/magicgammatest.csv",
      "delim" -> ",",
      "head" -> "false")

    val conf = new SparkConf().setAppName("Magicgamma").setMaster("local["+nCores+"]")

    conf.registerKryoClasses(Array(classOf[LSSVMSparkModel]))

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

    val (optModel, optConfig) = KernelizedModel.getOptimizedModel[RDD[(Long, LabeledPoint)],
      RDD[LabeledPoint], model.type](model, globalOptMethod,
        kernel, nProt, grid, step, logscale)

    optModel.setMaxIterations(2).learn()

    val met = optModel.evaluate(configtest)

    optModel.unpersist

    met.print()
    println("Optimal Configuration: "+optConfig)
    val scale = if(logscale) "log" else "linear"

    val perf = met.kpi()
    val row = Seq(kernel, prototypes.toString, globalOptMethod,
      grid.toString, step.toString, scale,
      perf(0), perf(1), perf(2), optConfig.toString)

    val writer = CSVWriter.open(new File("data/resultsMagicGamma.csv"), append = true)
    writer.writeRow(row)
    writer.close()

  }
}