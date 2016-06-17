package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar2812 17/6/16.
  *
  * Top level trait; represents the scaling operation, used
  * heavily in data processing tasks.
  */
trait Scaler[S] extends DataPipe[S, S]{
  def apply[T[S] <: Traversable[S]](data: T[S]) =
    data.map(run _).asInstanceOf[T[S]]
}

object Scaler {
  def apply[S](f: (S) => S): Scaler[S] =
    new Scaler[S] {
      override def run(data: S): S = f(data)
    }
}

/**
  * @author mandar2812 17/6/16
  *
  *
  * */
trait ReversibleScaler[S] extends Scaler[S] {

  def i: Scaler[S]
}