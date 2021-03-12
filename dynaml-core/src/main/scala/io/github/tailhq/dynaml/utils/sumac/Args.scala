package io.github.tailhq.dynaml.utils.sumac

import collection._

trait Args extends ExternalConfig with Serializable {
  def getArgs(argPrefix:String): Traversable[ArgAssignable] =
    getArgs(argPrefix, false, getDefaultArgs.map{a => a.getName -> a}.toMap)

  private[sumac] def nestedArgs: Vector[Args]

  private[sumac] def getArgs(argPrefix: String, gettingDefaults: Boolean, defaults: Map[String, ArgAssignable]): Traversable[ArgAssignable]

  /**
   * Returns the "default" values for the arguments of this class.  Unrelated to the current
   * value of those arguments.  Unless overridden, "default" means whatever values are assigned
   * by the no-arg constructor of this class.
   * @return
   */
  def getDefaultArgs: Traversable[ArgAssignable] = {
    try {
      this.getClass().newInstance().getArgs("", true, Map())
    } catch {
      case ie: InstantiationException => Traversable()  //nothing else we can do in this case, really
    }
  }

  @transient
  lazy val parser = ArgumentParser(getArgs(""))
  @Ignore
  @transient
  var validationFunctions: Seq[() => Unit] = Seq()

  def parse(commandLineArgs: String) {
    parse(ArgumentParser.argCLIStringToArgList(commandLineArgs))
  }

  def parse(args: Array[String]) {
    parse(ArgumentParser.argListToKvMap(args))
  }

  def parse(kvPairs: Map[String,String], validation: Boolean = true) {
    val modifiedKvPairs = if (validation) readArgs(kvPairs) else kvPairs
    val parsedArgs = parser.parse(modifiedKvPairs)
    parsedArgs.foreach { case (argAssignable, valueHolder) =>
      argAssignable.setValue(valueHolder.value)
    }
    if(kvPairs.contains("sumac.debugArgs")) {
      println("Sumac setup the following args:")
      getArgs("").foreach { arg =>
        println(s"\t--${arg.getName}\t${arg.getCurrentValue}")
      }
    }
    if (validation)
      runValidation()
  }

  override def readArgs(originalArgs: Map[String,String]): Map[String,String] = {
    originalArgs  //here for stackable traits
  }

  override def saveConfig() {} //noop, here for stackable traits

  /**
   * run all validation functions.
   *
   * Note that parse automatically runs all validation, so in general a user will not need to call this.  However,
   * if you are programatically filling in the the args of this object, you probably want to call this.
   */
  def runValidation() {
    validationFunctions.foreach{_()}
    nestedArgs.foreach{_.runValidation()}
  }

  def helpMessage = parser.helpMessage

  /**
   * add the ability to parse your own custom types.  Note that this
   * registers the parser *globally*, not just for this object.
   */
  def registerParser[T](parser: Parser[T]) {
    ParseHelper.registerParser(parser)
  }

  /**
   * get back the current (name, value) of all arguments as strings.  This does not just return the original arguments
   * that were passed to this function -- they may have been changed from defaulting, validation, etc.
   *
   * In general, users will not need this function, but it is useful for tools built on top, eg. saving to a property file
   */
  def getStringValues: Map[String,String] = {
    getArgs("").map{aa => aa.getName -> aa.getParser.valueAsString(aa.getCurrentValue, aa.getType)}.toMap
  }

  def addValidation(f:  => Unit) {
    validationFunctions ++= Seq(() => f)
  }
}
