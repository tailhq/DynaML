import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.kuleuven.esat.optimization.GridSearch
import org.kuleuven.esat.svm.LSSVMSparkModel

/**
 * @author mandar2812
 */

object TestMagicGamma {
  def apply(nCores: Int = 4, prototypes: Int = 1, kernel: String): Unit = {
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

    val nProt = if(prototypes > 0) prototypes else math.sqrt(model.npoints.toDouble).toInt
    //model.applyKernel(new RBFKernel(1.2), nProt)

    val gs = new GridSearch[RDD[(Long, LabeledPoint)], RDD[LabeledPoint], model.type](model)

    val (optModel, optConfig) = kernel match {
      case "RBF" => gs.optimize(Map("bandwidth" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "RBF", "subset" -> "100"))

      case "Polynomial" => gs.optimize(Map("degree" -> 1.0, "offset" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Polynomial", "subset" -> prototypes.toString))

      case "Exponential" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Exponential", "subset" -> prototypes.toString))

      case "Laplacian" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Laplacian", "subset" -> prototypes.toString))
    }

    optModel.setMaxIterations(2).learn()

    val met = optModel.evaluate(configtest)

    model.unpersist
    optModel.unpersist

    met.print()

  }
}