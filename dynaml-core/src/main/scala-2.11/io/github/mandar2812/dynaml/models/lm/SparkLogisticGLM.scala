package io.github.mandar2812.dynaml.models.lm

//Breeze Imports
import breeze.linalg.DenseVector
import breeze.numerics.sigmoid
import breeze.stats.distributions.Gaussian
import io.github.mandar2812.dynaml.optimization.ProbitGradient
import org.apache.spark.mllib.linalg.Vectors
//DynaML Imports
import io.github.mandar2812.dynaml.optimization.{
GradientDescentSpark, LogisticGradient,
RegularizedOptimizer, SquaredL2Updater}
//Spark Imports
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * @author mandar2812 date: 25/01/2017.
  *
  * Implementation of logistic regression model trained on Apache Spark [[RDD]]
  */
class SparkLogisticGLM(
  data: RDD[(DenseVector[Double], Double)], numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] =
  identity[DenseVector[Double]]) extends GenericGLM[
    RDD[(DenseVector[Double], Double)],
    RDD[LabeledPoint]](data, numPoints, map) {


  override def prepareData(d: RDD[(DenseVector[Double], Double)]) = d.map(pattern =>
      new LabeledPoint(
        pattern._2,
        Vectors.dense(featureMap(pattern._1).toArray ++ Array(1.0)))
  )

  private lazy val sample_input = g.first()._1

  /**
    * The link function; in this case simply the identity map
    * */
  override val h: (Double) => Double = sigmoid(_)

  override protected var params: DenseVector[Double] = initParams()

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double], DenseVector[Double],
    Double, RDD[LabeledPoint]] = new GradientDescentSpark(new LogisticGradient, new SquaredL2Updater)

  def dimensions = featureMap(sample_input).length

  override def initParams() = DenseVector.zeros[Double](dimensions + 1)

}

/**
  * @author mandar2812 date: 25/01/2017
  *
  * Implementation of probit GLM model trained on Apache Spark [[RDD]]
  * */
class SparkProbitGLM(
  data: RDD[(DenseVector[Double], Double)], numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] =
  identity[DenseVector[Double]]) extends SparkLogisticGLM(data, numPoints, map) {

  private val standardGaussian = new Gaussian(0, 1.0)

  override val h: (Double) => Double = (x: Double) => standardGaussian.cdf(x)

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double], DenseVector[Double],
    Double, RDD[LabeledPoint]] = new GradientDescentSpark(new ProbitGradient, new SquaredL2Updater)

}
