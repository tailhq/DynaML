import org.apache.spark.{SparkConf, SparkContext}
import org.kuleuven.esat.kernels.RBFKernel
import org.kuleuven.esat.svm.LSSVMSparkModel

/**
 * @author mandar2812
 */

val config = Map("file" -> "data/magicgamma.csv", "delim" -> ",",
  "head" -> "false",
  "task" -> "classification")

val configtest = Map("file" -> "data/magicgammatest.csv",
  "delim" -> ",",
  "head" -> "false")

val conf = new SparkConf().setAppName("Magicgamma").setMaster("local["+args(0).toInt+"]")

conf.registerKryoClasses(Array(classOf[LSSVMSparkModel]))

val sc = new SparkContext(conf)

val model = LSSVMSparkModel(config, sc)

model.applyKernel(new RBFKernel(100.0))
model.setRegParam(1.5).setLearningRate(0.001).setMaxIterations(5).learn()

val met = model.evaluate(configtest)

model.unpersist

met.print()

//met.generatePlots()
