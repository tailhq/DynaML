package io.github.mandar2812.dynaml.pipes

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
  * @author mandar2812 date 25/01/2017.
  *
  * Defines a light weight API for RDD operations
  * expressed as DynaML pipes.
  */
trait RDDPipe[Element, Result] extends DataPipe[RDD[Element], Result]

/**
  * A pipeline which takes an [[RDD]] of [[Element]] and maps it to
  * an [[RDD]] of [[OtherElement]].
  * */
trait RDDMapPipe[Element, OtherElement] extends RDDPipe[Element, RDD[OtherElement]] {
  self =>

  def >[Further](that: RDDPipe[OtherElement, Further]) = new RDDPipe[Element, Further] {
    override def run(data: RDD[Element]) = that.run(self.run(data))
  }

  def >[Further](that: RDDMapPipe[OtherElement, Further]) = new RDDMapPipe[Element, Further] {
    override def run(data: RDD[Element]) = that.run(self.run(data))
  }
}

trait RDDFilterPipe[Element] extends RDDPipe[Element, RDD[Element]]


object RDDPipe {

  def apply[E, R](f: (RDD[E]) => R) = new RDDPipe[E, R] {
    override def run(data: RDD[E]) = f(data)
  }

  def apply[E, R](f: (E) => R)(implicit c: ClassTag[R]) = new RDDMapPipe[E, R] {
    override def run(data: RDD[E]) = data.map(f)
  }

  def apply[E, R](f: (E) => Boolean) = new RDDFilterPipe[E] {
    override def run(data: RDD[E]) = data.filter(f)
  }

  def apply[E, R](p: StreamMapPipe[E, R])(implicit c: ClassTag[R]) = new RDDMapPipe[E, R] {
    override def run(data: RDD[E]) = data.mapPartitions(partition => p(partition.toStream).toIterator)
  }

}
