package io.github.mandar2812.dynaml.utils.sumac

import java.lang.reflect.Field
import java.lang.annotation.Annotation
import scala.collection._
import io.github.mandar2812.dynaml.utils.sumac.validation._

/**
 * Mix this trait into any class that you want to turn into an "argument holder".  It will automatically
 * convert each field of the class into a command line argument.  It will silently ignore fields that it doesn't
 * know how to parse.
 */
trait FieldArgs extends Args {
  private[sumac] override def getArgs(argPrefix:String, gettingDefaults: Boolean, defaults: Map[String, ArgAssignable]) = {
    val args: Seq[Seq[ArgAssignable]] = ReflectionUtils.getAllDeclaredFields(getClass) collect {
      case f: Field if (isValidField(f)) => {
        val fa = FieldArgAssignable(argPrefix, f, this)
        if(!gettingDefaults) addAnnotationValidations(fa, defaults)
        Seq(fa)
      }
      case nested: Field if (isNestedArgField(nested)) =>
        nested.setAccessible(true)
        val v = Option(nested.get(this)).getOrElse{
          val t = nested.getType.newInstance()
          nested.set(this, t)
          t
        }.asInstanceOf[Args]
        nestedArgs :+= v
        val subArgs: Seq[ArgAssignable] = v.getArgs(argPrefix + nested.getName + ".", gettingDefaults, defaults).toSeq
        subArgs
    }
    args.flatten
  }

  @Ignore
  private[sumac] var nestedArgs = Vector[Args]()

  def isSumacHelperField(f: Field): Boolean = f.getName == "parser" || f.getName == "bitmap$0"

  def isValidField(f: Field): Boolean = {
    ParseHelper.findParser(f.getType).isDefined && !isSumacHelperField(f) && hasSetter(f) && !f.isAnnotationPresent(classOf[Ignore])
  }

  def isNestedArgField(f: Field): Boolean = {
    classOf[Args].isAssignableFrom(f.getType)
  }

  def hasSetter(f: Field): Boolean = {
    //all fields in scala are private -- this is a way of checking if it has any public setter
    f.getDeclaringClass.getMethods.exists{_.getName() == f.getName + "_$eq"}
  }

  private[sumac] def addAnnotationValidations(f: FieldArgAssignable, defaultVals: Map[String, ArgAssignable]) {
    //Q: do inherited annotations mean anything on a field?  does it matter if I use getAnnotations vs getDeclaredAnnotations?
    f.field.getAnnotations.foreach { annot =>
      annotationValidationFunctions.get(annot.annotationType()).foreach{func =>
        val default = defaultVals(f.getName).getCurrentValue
        validationFunctions +:= new AnnotationValidationFunction(f, default, annot, func)
      }
    }
  }

  //just for debugging, nice to have a toString on this
  private class AnnotationValidationFunction(
    field: FieldArgAssignable,
    default: Any,
    annot: Annotation,
    check: (Any, Any, Annotation, String, ArgAssignable) => Unit
  ) extends (() => Unit) {
    def apply() {
      check(default, field.getCurrentValue, annot, field.getName, field)
    }
    override def toString() = "annotation check of " + field.getName
  }

  @transient
  private[sumac] val annotationValidationFunctions =
    mutable.Map[Class[_ <: Annotation], (Any,Any, Annotation, String, ArgAssignable) => Unit]()


  /**
   * Use this function to make an annotation automatically imply a validation function.  This registers the annotation
   * with *this* instance, so that any use of the annotation automatically adds the validation function on the field.
   *
   * In general, if you create user-defined annotation validation functions, you will want to call this in a base trait,
   * which all your arg classes extend, so you can use those annotations anywhere.
   *
   * @param annotation the class of the annotation to add a validation function to
   * @param validationFunction the function that will be called to validate every field marked w/ the annotation.  The
   *                           first argument is the default value of the argument, the second is the current value,
   *                           the third is the annotation, and the fourth is the name of the argument (for error msgs).
   */
  def registerAnnotationValidationUpdate(annotation: Class[_ <: Annotation])(validationFunction: (Any,Any, Annotation, String, ArgAssignable) => Unit) {
    annotationValidationFunctions += annotation -> validationFunction
  }

  def registerAnnotationValidation(annotation: Class[_ <: Annotation])(validationFunction: (Any,Any, Annotation, String) => Unit) {
    registerAnnotationValidationUpdate(annotation){(a:Any,b:Any,c:Annotation,d:String,e:ArgAssignable) => validationFunction(a,b,c,d)}
  }




  {
    //some built-in annotation validations
    registerAnnotationValidation(classOf[Required])(RequiredCheck)
    registerAnnotationValidation(classOf[Positive])(PositiveCheck)
    registerAnnotationValidation(classOf[Range])(RangeCheck)
    registerAnnotationValidation(classOf[FileExists])(FileExistsCheck)
  }

}

/**
 * Use this trait if you want an exception anytime your Argument class has a field that we don't know how to parse.
 * (FieldArgs just silently ignores those fields).
 *
 * Yes, the name is long -- if you want to use this as your standard way of parsing arguments, just alias it to a
 * shorter name in your projects.
 */
trait FieldArgsExceptionOnUnparseable extends FieldArgs {
  override def isValidField(f: Field): Boolean = {
    !isSumacHelperField(f) && hasSetter(f) && !f.isAnnotationPresent(classOf[Ignore])
  }
}