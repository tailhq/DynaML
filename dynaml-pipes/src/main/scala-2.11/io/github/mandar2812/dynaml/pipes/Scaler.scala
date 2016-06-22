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

  def *[T](that: Scaler[T]) = {
    val firstRun = this.run _
    new Scaler[(S,T)] {
      override def run(data: (S, T)): (S, T) = (firstRun(data._1), that(data._2))
    }
  }

  def >(otherScaler: Scaler[S]) = {

    val firstRun = this.run _

    new Scaler[S] {
      def run(data: S) = otherScaler.run(firstRun(data))
    }
  }

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

  /**
    * The inverse operation of this scaling.
    *
    * */
  val i: Scaler[S]

  def *[T](that: ReversibleScaler[T]) = {

    val firstInv = this.i

    val firstRun = this.run _

    new ReversibleScaler[(S, T)] {

      val i: Scaler[(S,T)] = firstInv * that.i

      override def run(data: (S, T)): (S, T) = (firstRun(data._1), that(data._2))
    }
  }

  def >(otherRevScaler: ReversibleScaler[S]): ReversibleScaler[S] = {

    val firstInv = this.i

    val firstRun = this.run _

    new ReversibleScaler[S] {
      val i: Scaler[S] = otherRevScaler.i > firstInv
      def run(data: S) = otherRevScaler.run(firstRun(data))
    }
  }
}