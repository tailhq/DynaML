import org.apache.spark.{SparkConf, SparkContext}
import org.kuleuven.esat.svm.LSSVMSparkModel

/**
 * @author mandar2812
 */

val config = Map("file" -> "data/ripley.csv", "delim" -> ",",
  "head" -> "false",
  "task" -> "classification")

val configtest = Map("file" -> "data/ripleytest.csv",
  "delim" -> ",",
  "head" -> "false")

val conf = new SparkConf().setAppName("Magicgamma").setMaster("local[4]")

conf.registerKryoClasses(Array(classOf[LSSVMSparkModel]))

val sc = new SparkContext(conf)

val model = LSSVMSparkModel(config, sc)

model.setRegParam(0.5).setLearningRate(0.001).setMaxIterations(10).learn()

val met = model.evaluate(configtest)

met.print()

met.generatePlots()
