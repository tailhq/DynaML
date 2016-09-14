package io.github.mandar2812.dynaml.dataformat

import scala.collection.JavaConversions

/**
  * Created by mandar on 08/09/16.
  */
object ARFF {
  def asStream(f: ArffFile): Stream[String] =
    JavaConversions.asScalaBuffer(f.getData).toStream.map(_.map(_.toString).mkString(","))
}
