package io.github.tailhq.dynaml.utils.sumac

import io.github.tailhq.dynaml.utils.sumac.validation.Required

import scala.annotation.tailrec
import java.lang.reflect.{Type, Field}
import collection.mutable.LinkedHashMap
import collection._

class ArgumentParser[T <: ArgAssignable] (val argHolders: Seq[T]) {
  lazy val nameToHolder:Map[String,T] = (LinkedHashMap.empty ++ argHolders.map(a => a.getName -> a)).withDefault { arg =>
    throw new ArgException("unknown option %s\n%s".format(arg, helpMessage))
  }

  def parse(commandLineArgs: String): Map[T, ValueHolder[_]] = {
    parse(ArgumentParser.argCLIStringToArgList(commandLineArgs))
  }

  def parse(args: Array[String]): Map[T, ValueHolder[_]] = {
    parse(ArgumentParser.argListToKvMap(args))
  }

  def parse(rawKvs: Map[String,String]): Map[T, ValueHolder[_]] = {
    if (rawKvs.contains("help")) {
      throw new FeedbackException(helpMessage)
    }
    rawKvs.collect {
      case(argName, argValue) if !ArgumentParser.isReserved(argName) =>
        val holder = nameToHolder(argName)
        val result = try {
          ParseHelper.parseInto(argValue, holder.getType, holder.getCurrentValue) getOrElse {
            throw new FeedbackException("don't know how to parse type: " + holder.getType)
          }
        } catch {
          case ae: ArgException => throw ae
          case e: Throwable => throw new ArgException("Error parsing \"%s\" into field \"%s\" (type = %s)\n%s".format(argValue, argName, holder.getType, helpMessage), e)
        }
        holder -> result
    }
  }

  def helpMessage: String = {
    val msg = new StringBuilder
    msg.append("usage: \n")
    nameToHolder.foreach { case (k, v) =>
      msg.append(v.toString() + "\n")
    }
    msg.toString
  }
}

object ArgumentParser {

  val reservedArguments = Seq("help", "sumac.debugArgs")
  // backslash follower by newline for newline, carriage return, and combinations
  // Newlines have to come first!
  val newlineCharacters = Seq("\\\n", "\\\r", "\\\n\r", "\\\r\n")

  def isReserved(name: String) = reservedArguments.contains(name)

  def apply[T <: ArgAssignable](argHolders: Traversable[T]) = {
    // ignore things we don't know how to parse
    new ArgumentParser(argHolders.toSeq.filter(t => ParseHelper.findParser(t.getType).isDefined))
  }

  def argCLIStringToArgList(commandLineArgs: String): Array[String] = {

    /**
     * Helper method for preserving quoted strings while splitting on whitespace
     */
    def splitRespectingQuotes(s: String): Array[String] = {
      var openQuote: Option[Char] = None
      var stringBuilder: StringBuilder = new StringBuilder
      var splitArray: Array[String] = Array()
      var escaping: Boolean = false

      // Crawl through string character-by-character
      s.foreach{case(char) => {
        char match {
          case '\\' => {
            // If encounter backslash (escape character) keep note for checking next character
            // Append two backslashes if we encounter two backslashes in a row.
            if(escaping) {
              stringBuilder ++= "\\\\"
            }
            escaping = !escaping
          }
          case s if "\\s".r.findFirstIn(s.toString).isDefined && openQuote.isEmpty => {
            // If we encounter whitespace and aren't inside of a quote
            // add this string to the array and start a new string
            if(stringBuilder.size > 0) {
              splitArray = splitArray ++ Array(stringBuilder.toString())
              stringBuilder = new StringBuilder
            }
          }
          case s if openQuote == Some(s) => {
            // If we are closing an open quote (by matching " to " or ' to ')
            // First check if its escaped, otherwise end this quote block.
            if(escaping) {
              escaping = false
              stringBuilder ++= "\\\""
            } else {
              openQuote = None
              splitArray = splitArray ++ Array(stringBuilder.toString())
              stringBuilder = new StringBuilder
            }
          }
          case s if s == '"' || s == '\'' && openQuote.isEmpty && !escaping => {
            // check if we are encountering an open quote that isn't escaped
            openQuote = Some(s)
          }
          case _ => {
            // If no other condition is met, append the character, and append a backslash if we previously saw one
            if(escaping) {
              stringBuilder += '\\'
              escaping = !escaping
            }
            stringBuilder += char
          }
        }
      }}

      if(stringBuilder.isEmpty) splitArray else splitArray ++ Array(stringBuilder.toString)
    }

    val removeNewlines = newlineCharacters.foldLeft[String](commandLineArgs){case(currentArgString, character) =>
      currentArgString.replaceAllLiterally(character, "")
    }.trim()
    splitRespectingQuotes(removeNewlines)
  }

  def argListToKvMap(args: Array[String]): Map[String,String] = {
    @tailrec
    def parse(args: List[String], acc: mutable.Map[String, String] = mutable.Map.empty): mutable.Map[String,String] = {
      args match {
        case Nil => acc
        case "--help" :: _ =>
          acc("help") = null
          acc
        case arg :: _ if (!arg.startsWith("--")) =>
          throw new FeedbackException("expecting argument name beginning with \"--\", instead got %s".format(arg))
        case name :: value :: tail =>
          val suffix = name.drop(2)
          acc(suffix) = value
          parse(tail, acc)
        case _ => throw new FeedbackException("gave a non-key value argument")
      }
    }
    parse(args.toList)
  }
}

/**
 * Container for one argument, that has name, type, and can be assigned a value.
 */
trait ArgAssignable {
  def getName: String
  def getDescription: String
  def getType: Type
  def getCurrentValue: AnyRef
  def getParser: Parser[_]
  def setValue(value: Any)

  def allowedValues: Option[Set[String]] = getParser.allowedValues(getType, getCurrentValue)
  def required: Boolean = false

  override def toString() = {
    var t = "--" + getName + "\t" + getType
    if (getDescription != getName)
      t += "\t" + getDescription
    t += "\t" + getCurrentValue
    t
  }
}

class FieldArgAssignable(val prefix: String, val field: Field, val obj: Object, val parser: Parser[_]) extends ArgAssignable {
  field.setAccessible(true)
  val annotationOpt = Option(field.getAnnotation(classOf[Arg]))
  override val required = field.getAnnotation(classOf[Required]) != null
  def getParser = parser

  def getName = {
    prefix + {
      val n = annotationOpt.map(_.name).getOrElse(field.getName)
      if (n == "") field.getName else n
    }
  }

  def getDescription = {
    val d = annotationOpt.map(_.description).getOrElse(field.getName)
    if (d == "") getName else d
  }

  def getType = field.getGenericType
  def getCurrentValue = field.get(obj)

  def setValue(value: Any) = {
    field.set(obj, value)
  }
}

object FieldArgAssignable{
  def apply(argPrefix: String, field: Field, obj: Object): FieldArgAssignable = {
    val tpe = field.getGenericType
    val parser = ParseHelper.findParser(tpe) getOrElse {
      throw new ArgException("don't know how to parse type: " + tpe)
    }
    new FieldArgAssignable(argPrefix, field, obj, parser)
  }
}

class ArgException(msg: String, cause: Throwable) extends IllegalArgumentException(msg, cause) {
  def this(msg: String) = this(msg, null)
}

object ArgException {
  def apply(msg: String, cause: Throwable) = new ArgException(msg, cause)
}

class FeedbackException(msg: String) extends ArgException(msg, null)


