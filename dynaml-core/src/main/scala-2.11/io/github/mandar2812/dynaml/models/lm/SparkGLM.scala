package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.optimization.{RegularizedLSSolver, RegularizedOptimizer}
import org.apache.spark.rdd.RDD

/**
  * @author mandar2812 date: 25/01/2017.
  * A Generalized Linear Model applied to a
  * single output regression task. The training
  * set is an Apache Spark [[RDD]]
  *
  * @param data The training data as an [[RDD]]
  * @param numPoints Number of training data points
  * @param map A general non-linear feature mapping/basis function expansion.
  *
  */
class SparkGLM(
  data: RDD[(DenseVector[Double], Double)], numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]])
  extends GenericGLM[
    RDD[(DenseVector[Double], Double)],
    (DenseMatrix[Double], DenseVector[Double])](data, numPoints, map) {

  private lazy val sample_input = g.first()._1

  /**
    * The link function; in this case simply the identity map
    * */
  override val h: (Double) => Double = identity[Double]

  featureMap = map

  override protected var params: DenseVector[Double] = initParams()

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double], DenseVector[Double],
    Double, (DenseMatrix[Double], DenseVector[Double])] = new RegularizedLSSolver

  def dimensions = featureMap(sample_input).length

  override def initParams() = DenseVector.zeros[Double](dimensions + 1)

  /**
    * Input an [[RDD]] containing the data set and output
    * a design matrix and response vector which can be solved
    * in the OLS sense.
    * */
  override def prepareData(d: RDD[(DenseVector[Double], Double)]) = {

    val phi = featureMap
    val mapFunc = (xy: (DenseVector[Double], Double)) => {
      val phiX = DenseVector(phi(xy._1).toArray ++ Array(1.0))
      val phiY = phiX*xy._2
      (phiX*phiX.t, phiY)
    }

    d.mapPartitions((partition) => {
      Iterator(partition.map(mapFunc).reduce((a,b) => (a._1+b._1, a._2+b._2)))
    }).reduce((a,b) => (a._1+b._1, a._2+b._2))
  }
}
