package io.github.mandar2812.dynaml.utils.sumac.validation

import io.github.mandar2812.dynaml.utils.sumac.{FeedbackException, ArgException}
import java.lang.annotation.Annotation
import java.io.File

object RequiredCheck extends Function4[Any, Any, Annotation, String, Unit] {

  def apply(defaultValue: Any, currentValue: Any, annot: Annotation, name: String) {
    if (defaultValue == currentValue)
      throw new FeedbackException("must specify a value for " + name)
  }

  override def toString() = getClass().getSimpleName
}


object PositiveCheck extends Function4[Any, Any, Annotation, String, Unit] {
  def apply(defaultValue: Any, currentValue: Any, annot: Annotation, name: String) {
    numericAsDouble(currentValue){v =>
      if (v <= 0.0)
        throw new FeedbackException("must specify a positive value for " + name)
    }
  }

  def numericAsDouble[T](v: Any)(f: Double => T): Option[T] = {
    v match {
      case i: Int => Some(f(i.toDouble))
      case l: Long => Some(f(l.toDouble))
      case fl: Float => Some(f(fl.toDouble))
      case d: Double => Some(f(d))
      case _ => None
    }
  }

  override def toString() = getClass().getSimpleName
}

object RangeCheck extends Function4[Any, Any, Annotation, String, Unit] {
  def apply(defaultValue: Any, currentValue: Any, annot: Annotation, name: String) {
    annot match {
      case r:Range =>
        PositiveCheck.numericAsDouble(currentValue){v =>
          if (v < r.min() || v > r.max())
            throw new FeedbackException("must specify a value between " + r.min() + " and " + r.max() + " for " + name)
        }
      case _ => ()
    }
  }
}

object FileExistsCheck extends Function4[Any, Any, Annotation, String, Unit] {


  def apply(defaultValue: Any, currentValue: Any, annot: Annotation, name: String) {
    Option(currentValue).map{asFile(_)} match {
      case Some(f) if !f.exists() =>
        throw new FeedbackException("must specify a file that exists for %s, current value = %s".format(name, f.toString))
      case None => throw new FeedbackException("must specify a valid file name for " + name)
      case _ => // Valid case
    }
  }

  def asFile(v: Any): File = {
    v match {
      case s: String => new File(s)
      case f: File => f
    }
  }

}