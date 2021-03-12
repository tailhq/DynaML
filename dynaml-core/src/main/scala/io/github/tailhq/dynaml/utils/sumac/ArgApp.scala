package io.github.tailhq.dynaml.utils.sumac

trait Argable[T <: Args] {

  protected lazy val argHolder = {
    val argClass = getArgumentClass
    ReflectionUtils.construct[T](argClass)
  }

  /**
   * get the instance of T that holds the parsed args.
   *
   * not needed for the user that just wants to run their code -- this is accessible just for other libs
   * built on top.
   */
  def getArgHolder: T = argHolder

  private[sumac] def getArgumentClass: Class[T] = {
    //we need to get the type parameter for Argable.  Doing that requires searching through the interfaces of *all*
    // classes in the type hierarchy.
    val argApp = ReflectionUtils.findGenericInterface(getClass, classOf[Argable[_]])
    ReflectionUtils.getRawClass(argApp.get.getActualTypeArguments.apply(0)).asInstanceOf[Class[T]]
  }
}

trait ArgMain[T <: FieldArgs] extends Argable[T] {
  def main(rawArgs: Array[String]) {
    mainHelper(rawArgs)
  }

  private def mainHelper(rawArgs: Array[String]) {
    try {
      argHolder.parse(rawArgs)
    } catch {
      case ex: FeedbackException =>
        println(ex.getMessage)
        System.exit(1)
    }
    main(argHolder)
  }

  def main(args: T)
}

trait ArgFunction[T <: FieldArgs, U] extends Function[T, U] with Argable[T]

@deprecated("you should avoid using this until a replacement to DelayedInit has been introduced to scala.", "24/04/2014")
trait ArgApp[T <: FieldArgs] extends Argable[T] with App {
  override def main(args: Array[String]) {
    try {
      argHolder.parse(args)
    } catch {
      case ex: FeedbackException =>
        println(ex.getMessage)
        System.exit(1)
    }
    super.main(args)
  }
}

//below is just for testing, but want it in compiled classes ...

class MyArgs extends FieldArgs {
  var a: String = ""
  var b: Int = 0
}

object MyMain extends ArgMain[MyArgs] {
  def main(args: MyArgs) {
    println(args.a)
    println(args.b)
  }
}
