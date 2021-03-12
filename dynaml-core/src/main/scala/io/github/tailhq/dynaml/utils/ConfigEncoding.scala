package io.github.tailhq.dynaml.utils

import breeze.linalg.DenseVector
import io.github.tailhq.dynaml.pipes.{DataPipe, Encoder}

/**
  * An encoding which converts a hyper-parameter configuration
  * from a [[Map]] to a breeze [[DenseVector]] and back.
  *
  * @param keys A list of hyper-parameter strings.
  *
  * */
case class ConfigEncoding(keys: List[String]) extends Encoder[Map[String, Double], DenseVector[Double]] {
  self =>

  override val i = DataPipe((x: DenseVector[Double]) => keys.zip(x.toArray).toMap)

  override def run(data: Map[String, Double]) = DenseVector(keys.map(data(_)).toArray)

  def reverseEncoder = Encoder(i, DataPipe(self.run _))
}
