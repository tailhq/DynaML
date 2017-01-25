package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.optimization.{RegularizedLSSolver, RegularizedOptimizer}
import org.apache.spark.rdd.RDD

/**
  * @author mandar2812 date: 25/01/2017.
  * A Generalized Linear Model applied to a
  * single output regression task. The training
  * set is an Apache Spark [[RDD]]
  */
class SparkGLM(
  data: RDD[(DenseVector[Double], Double)], numPoints: Int,
  map: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]],
  num_features: Int = -1)
  extends GenericGLM[
    RDD[(DenseVector[Double], Double)],
    (DenseMatrix[Double], DenseVector[Double])](data, numPoints, map) {

  private lazy val sample_input = g.first()._1

  override val h: (Double) => Double = identity[Double]

  override protected var params: DenseVector[Double] = initParams()

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double], DenseVector[Double],
    Double, (DenseMatrix[Double], DenseVector[Double])] = new RegularizedLSSolver

  def dimensions = if(num_features > 0) num_features else featureMap(sample_input).length

  override def initParams() = DenseVector.zeros[Double](dimensions)

  override def prepareData(d: RDD[(DenseVector[Double], Double)]) = {

    val phi = featureMap
    val mapFunc = (xy: (DenseVector[Double], Double)) => {
      val phiX = phi(xy._1)
      val phiY = phi(xy._1)*xy._2
      (phiX*phiX.t, phiY)
    }

    d.mapPartitions((partition) => {
      Iterator(partition.map(mapFunc).reduce((a,b) => (a._1+b._1, a._2+b._2)))
    }).reduce((a,b) => (a._1+b._1, a._2+b._2))
  }
}
