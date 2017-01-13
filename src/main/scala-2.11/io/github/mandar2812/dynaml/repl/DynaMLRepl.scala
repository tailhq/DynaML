package io.github.mandar2812.dynaml.repl

import java.io.{InputStream, OutputStream}

import ammonite.util.Ref
import ammonite.ops.Path
import ammonite.repl.Repl
import ammonite.runtime.Storage
import ammonite.util.Bind

import scala.io.Source
import ammonite.runtime.ImportHook
import ammonite.ops._
import ammonite.util.Name.backtickWrap
import ammonite.util.{Res, Util}
import fastparse.utils.Utils._
import io.github.mandar2812.dynaml.repl.Router.{ArgSig, EntryPoint}
import sourcecode.Compat.Context

import scala.annotation.StaticAnnotation
import scala.language.experimental.macros


/**
  * Created by mandar on 1/6/16.
  */
class DynaMLRepl(input: InputStream,
                 output: OutputStream,
                 error: OutputStream,
                 storage: Ref[Storage],
                 predef: String = "",
                 replArgs: Seq[Bind[_]] = Nil,
                 promptStr: String,
                 wd: ammonite.ops.Path,
                 welcomeBanner: Option[String]) extends
  Repl(input, output, error, storage(), predef, wd, welcomeBanner, replArgs) {

  override val prompt = Ref("DynaML>")

}

object Defaults{

  val welcomeBanner = {
    def ammoniteVersion = ammonite.Constants.version
    def scalaVersion = scala.util.Properties.versionNumberString
    def javaVersion = System.getProperty("java.version")
    def version = BuildInfo.version
    def banner = Source.fromFile("./conf/banner.txt").getLines.mkString("\n")

    Util.normalizeNewlines(
      banner+s"""\nWelcome to DynaML $version \nInteractive Scala shell for Machine Learning Research
          |
          |Currently running on:
          |(Scala $scalaVersion Java $javaVersion)
          |""".stripMargin
    )
  }

  val ignoreUselessImports = """
                               |notify => _,
                               |  wait => _,
                               |  equals => _,
                               |  asInstanceOf => _,
                               |  synchronized => _,
                               |  notifyAll => _,
                               |  isInstanceOf => _,
                               |  == => _,
                               |  != => _,
                               |  getClass => _,
                               |  ne => _,
                               |  eq => _,
                               |  ## => _,
                               |  hashCode => _,
                               |  _
                               |"""

  def dynaMlPredef = Source.fromFile("./conf/DynaMLInit.scala").getLines.mkString("\n")

  val predefString = s"""
                        |import ammonite.ops.Extensions.{
                        |  $ignoreUselessImports
                        |}
                        |import ammonite.runtime.tools._
                        |import ammonite.repl.tools._
                        |import ammonite.runtime.tools.IvyConstructor.{ArtifactIdExt, GroupIdExt}
                        |import ammonite.repl.ReplBridge.value.{exit, codeColors, tprintColors, show, typeOf}
                        |import io.github.mandar2812.dynaml.repl.Router.{doc, main}
                        |import io.github.mandar2812.dynaml.repl.Scripts.pathScoptRead
                        |""".stripMargin + dynaMlPredef


  def ammoniteHome = ammonite.ops.Path(System.getProperty("user.home"))/".ammonite"

}



object Router{
  class doc(s: String) extends StaticAnnotation
  class main extends StaticAnnotation
  def generateRoutes[T](t: T): Seq[Router.EntryPoint] = macro generateRoutesImpl[T]
  def generateRoutesImpl[T: c.WeakTypeTag](c: Context)(t: c.Expr[T]): c.Expr[Seq[EntryPoint]] = {
    import c.universe._
    val r = new Router(c)
    val allRoutes = r.getAllRoutesForClass(
      weakTypeOf[T].asInstanceOf[r.c.Type],
      t.tree.asInstanceOf[r.c.Tree]
    ).asInstanceOf[Iterable[c.Tree]]

    val res = q"_root_.scala.Seq(..$allRoutes)"
    //    println(res)
    c.Expr[Seq[EntryPoint]](res)
  }

  /**
    * Models what is known by the router about a single argument: that it has
    * a [[name]], a human-readable [[typeString]] describing what the type is
    * (just for logging and reading, not a replacement for a `TypeTag`) and
    * possible a function that can compute its default value
    */
  case class ArgSig(name: String,
                    typeString: String,
                    doc: Option[String],
                    default: Option[() => Any])

  /**
    * What is known about a single endpoint for our routes. It has a [[name]],
    * [[argSignatures]] for each argument, and a macro-generated [[invoke0]]
    * that performs all the necessary argument parsing and de-serialization.
    *
    * Realistically, you will probably spend most of your time calling [[invoke]]
    * instead, which provides a nicer API to call it that mimmicks the API of
    * calling a Scala method.
    */
  case class EntryPoint(name: String,
                        argSignatures: Seq[ArgSig],
                        varargs: Boolean,
                        invoke0: (Map[String, String], Seq[String]) => Result[Any]){
    def invoke(args: Seq[String], kwargs: Seq[(String, String)]): Result[Any] = {
      val (usedArgs, leftoverArgs) = args.splitAt(argSignatures.length)
      if (leftoverArgs.nonEmpty && !varargs) Result.Error.TooManyArguments(leftoverArgs)
      else {
        val implicitlyNamedArgs = argSignatures.map(_.name).zip(usedArgs).toMap
        val redundantKeys =
          (implicitlyNamedArgs.keys.toSeq ++ kwargs.map(_._1))
            .groupBy(x=>x)
            .filter(_._2.size > 1)

        if(redundantKeys.nonEmpty) {
          Result.Error.RedundantArguments(redundantKeys.keysIterator.toSeq)
        } else {
          try invoke0(implicitlyNamedArgs ++ kwargs, leftoverArgs)
          catch{case e: Throwable =>
            Result.Error.Exception(e)
          }
        }
      }
    }
  }

  def read[T](dict: Map[String, String],
              default: => Option[Any],
              arg: ArgSig,
              thunk: String => T,
              extras: Option[Seq[String]]): FailMaybe = {
    dict.get(arg.name) match{
      case None =>
        try default match{
          case None => Left(Seq(Result.ParamError.Missing(arg)))
          case Some(v) => Right(v)
        } catch {case e => Left(Seq(Result.ParamError.DefaultFailed(arg, e))) }

      case Some(x) =>

        extras match{
          case None =>
            try Right(thunk(x))
            catch {case e => Left(Seq(Result.ParamError.Invalid(arg, x, e))) }

          case Some(extraItems) =>
            val attempts: Seq[Either[Result.ParamError.Invalid, T]] = (x +: extraItems).map{ item =>
              try Right(thunk(item))
              catch {case e => Left(Result.ParamError.Invalid(arg, item, e)) }
            }

            val bad = attempts.collect{ case Left(x) => x}
            if (bad.nonEmpty) Left(bad)
            else {
              val good = Right(attempts.collect{case Right(x) => x})
              good
            }
        }

    }
  }

  /**
    * Represents what comes out of an attempt to invoke an [[EntryPoint]].
    * Could succeed with a value, but could fail in many different ways.
    */
  sealed trait Result[+T]
  object Result{

    /**
      * Invoking the [[EntryPoint]] was totally successful, and returned a
      * result
      */
    case class Success[T](value: T) extends Result[T]

    /**
      * Invoking the [[EntryPoint]] was not successful
      */
    sealed trait Error extends Result[Nothing]
    object Error{

      /**
        * Invoking the [[EntryPoint]] failed with an exception while executing
        * code within it.
        */
      case class Exception(t: Throwable) extends Error
      /**
        * Invoking the [[EntryPoint]] failed as there were too many positional
        * arguments passed in, more than what is expected by the [[EntryPoint]]
        */
      case class TooManyArguments(values: Seq[String]) extends Error
      /**
        * Invoking the [[EntryPoint]] failed as the same argument was passed
        * in more than once; possibly as a keyword-argument that was repeated,
        * or a keyword-argument and positional-argument that both resolve to
        * the same arg
        */
      case class RedundantArguments(names: Seq[String]) extends Error

      /**
        * Invoking the [[EntryPoint]] failed because there were problems
        * deserializing/parsing individual arguments
        */
      case class InvalidArguments(values: Seq[ParamError]) extends Error
    }

    /**
      * What could go wrong when trying to parse an individual parameter to
      * an [[EntryPoint]]?
      */
    sealed trait ParamError
    object ParamError {
      /**
        * Some parameter was missing from the input.
        */
      case class Missing(arg: ArgSig) extends ParamError
      /**
        * Something went wrong trying to de-serialize the input parameter;
        * the thrown exception is stored in [[ex]]
        */
      case class Invalid(arg: ArgSig, value: String, ex: Throwable) extends ParamError
      /**
        * Something went wrong trying to evaluate the default value
        * for this input parameter
        */
      case class DefaultFailed(arg: ArgSig, ex: Throwable) extends ParamError
    }
  }


  type FailMaybe = Either[Seq[Result.ParamError], Any]
  type FailAll = Either[Seq[Result.ParamError], Seq[Any]]

  def validate(args: Seq[FailMaybe]): Result[Seq[Any]] = {
    val lefts = args.collect{case Left(x) => x}.flatten
    if (lefts.nonEmpty) Result.Error.InvalidArguments(lefts)
    else Result.Success(args.collect{case Right(x) => x})
  }
}
class Router [C <: Context](val c: C) {
  import c.universe._
  def getValsOrMeths(curCls: Type): Iterable[MethodSymbol] = {
    def isAMemberOfAnyRef(member: Symbol) =
      weakTypeOf[AnyRef].members.exists(_.name == member.name)
    val extractableMembers = for {
      member <- curCls.declarations
      if !isAMemberOfAnyRef(member)
      if !member.isSynthetic
      if member.isPublic
      if member.isTerm
      memTerm = member.asTerm
      if memTerm.isMethod
    } yield memTerm.asMethod
    extractableMembers flatMap { case memTerm =>
      if (memTerm.isSetter || memTerm.isConstructor || memTerm.isGetter) Nil
      else Seq(memTerm)

    }
  }

  def extractMethod(meth: MethodSymbol,
                    curCls: c.universe.Type,
                    target: c.Tree): c.universe.Tree = {
    val flattenedArgLists = meth.paramss.flatten
    def hasDefault(i: Int) = {
      val defaultName = s"${meth.name}$$default$$${i + 1}"
      if (curCls.members.exists(_.name.toString == defaultName)) {
        Some(defaultName)
      } else {
        None
      }
    }
    val argListSymbol = q"${c.fresh[TermName]("argsList")}"
    val defaults = for ((arg, i) <- flattenedArgLists.zipWithIndex) yield {
      hasDefault(i).map(defaultName => q"() => $target.${newTermName(defaultName)}")
    }

    def unwrapVarargType(arg: Symbol) = {
      val vararg = arg.typeSignature.typeSymbol == definitions.RepeatedParamClass
      val unwrappedType =
        if (!vararg) arg.typeSignature
        else arg.typeSignature.asInstanceOf[TypeRef].args(0)

      (vararg, unwrappedType)
    }
    val readArgSigs = for(
      ((arg, defaultOpt), i) <- flattenedArgLists.zip(defaults).zipWithIndex
    ) yield {

      val (vararg, unwrappedType) = unwrapVarargType(arg)

      val default =
        if (vararg) q"scala.Some(scala.Nil)"
        else defaultOpt match {
          case Some(defaultExpr) => q"scala.Some($defaultExpr())"
          case None => q"scala.None"
        }

      val docs = for{
        doc <- arg.annotations.find(_.tpe =:= typeOf[Router.doc])
        if doc.scalaArgs.head.isInstanceOf[Literal]
        l =  doc.scalaArgs.head.asInstanceOf[Literal]
        if l.value.value.isInstanceOf[String]
      } yield l.value.value.asInstanceOf[String]

      val docTree = docs match{
        case None => q"scala.None"
        case Some(s) => q"scala.Some($s)"
      }
      val argSig = q"""
        Router.ArgSig(
          ${arg.name.toString},
          ${arg.typeSignature.toString},
          $docTree,
          $defaultOpt
        )
      """

      val extraArg = if(vararg) q"scala.Some(extras)" else q"None"
      val reader = q"""
      Router.read[$unwrappedType](
        $argListSymbol,
        $default,
        $argSig,
        implicitly[scopt.Read[$unwrappedType]].reads(_),
        $extraArg
      )
      """
      (reader, argSig, vararg)
    }

    val (readArgs, argSigs, varargs) = readArgSigs.unzip3
    val (argNames, argNameCasts) = flattenedArgLists.map { arg =>
      val (vararg, unwrappedType) = unwrapVarargType(arg)
      (
        pq"${arg.name.toTermName}",
        if (!vararg) q"${arg.name.toTermName}.asInstanceOf[$unwrappedType]"
        else q"${arg.name.toTermName}.asInstanceOf[Seq[$unwrappedType]]: _*"

        )
    }.unzip

    q"""
    Router.EntryPoint(
      ${meth.name.toString},
      scala.Seq(..$argSigs),
      ${varargs.contains(true)},
      ($argListSymbol: Map[String, String], extras: Seq[String]) =>
        Router.validate(Seq(..$readArgs)) match{
          case Router.Result.Success(List(..$argNames)) =>
            Router.Result.Success($target.${meth.name.toTermName}(..$argNameCasts))
          case x => x
        }
    )
    """
  }

  def getAllRoutesForClass(curCls: Type, target: c.Tree): Iterable[c.universe.Tree] = for{
    t <- getValsOrMeths(curCls)
    if t.annotations.exists(_.tpe =:= typeOf[Router.main])
  } yield extractMethod(t, curCls, target)
}



object Scripts {
  def runScript(wd: Path,
                path: Path,
                interp: ammonite.interp.Interpreter,
                args: Seq[String],
                kwargs: Seq[(String, String)]) = {
    val (pkg, wrapper) = Util.pathToPackageWrapper(path, wd)
    for{
      (imports, wrapperHashes) <- interp.processModule(
        ImportHook.Source.File(path),
        Util.normalizeNewlines(read(path)),
        wrapper,
        pkg,
        autoImport = true,
        // Not sure why we need to wrap this in a separate `$routes` object,
        // but if we don't do it for some reason the `generateRoutes` macro
        // does not see the annotations on the methods of the outer-wrapper.
        // It can inspect the type and its methods fine, it's just the
        // `methodsymbol.annotations` ends up being empty.
        extraCode = Util.normalizeNewlines(
          s"""
             |val $$routesOuter = this
             |object $$routes extends scala.Function0[scala.Seq[Router.EntryPoint]]{
             |  def apply() = Router.generateRoutes[$$routesOuter.type]($$routesOuter)
             |}
          """.stripMargin
        )
      )

      routeClsName = wrapperHashes.last._1

      routesCls =
      interp
        .eval
        .frames
        .head
        .classloader
        .loadClass(routeClsName + "$$routes$")

      scriptMains =
      routesCls
        .getField("MODULE$")
        .get(null)
        .asInstanceOf[() => Seq[Router.EntryPoint]]
        .apply()

      res <- scriptMains match {
        // If there are no @main methods, there's nothing to do
        case Seq() => Res.Success(imports)
        // If there's one @main method, we run it with all args
        case Seq(main) => runMainMethod(main, args, kwargs).getOrElse(Res.Success(imports))
        // If there are multiple @main methods, we use the first arg to decide
        // which method to run, and pass the rest to that main method
        case mainMethods =>
          val suffix = formatMainMethods(mainMethods)
          args match{
            case Seq() =>
              Res.Failure(
                None,
                s"Need to specify a main method to call when running " + path.last + suffix
              )
            case Seq(head, tail @ _*) =>
              mainMethods.find(_.name == head) match{
                case None =>
                  Res.Failure(
                    None,
                    s"Unable to find method: " + backtickWrap(head) + suffix
                  )
                case Some(main) =>
                  runMainMethod(main, tail, kwargs).getOrElse(Res.Success(imports))
              }
          }
      }
    } yield res
  }
  def formatMainMethods(mainMethods: Seq[Router.EntryPoint]) = {
    if (mainMethods.isEmpty) ""
    else{
      val methods = for(main <- mainMethods) yield{
        val args = main.argSignatures.map(renderArg).mkString(", ")
        val details = mainMethodDetails(main)
        s"def ${main.name}($args)$details"
      }
      Util.normalizeNewlines(
        s"""
           |
           |Available main methods:
           |
           |${methods.mkString(Util.newLine)}""".stripMargin
      )
    }
  }
  def runMainMethod(mainMethod: Router.EntryPoint,
                    args: Seq[String],
                    kwargs: Seq[(String, String)]): Option[Res.Failing] = {

    def expectedMsg = {
      val commaSeparated =
        mainMethod.argSignatures
          .map(renderArg)
          .mkString(", ")
      val details = mainMethodDetails(mainMethod)
      "(" + commaSeparated + ")" + details
    }

    mainMethod.invoke(args, kwargs) match{
      case Router.Result.Success(x) => None
      case Router.Result.Error.Exception(x) => Some(Res.Exception(x, ""))
      case Router.Result.Error.TooManyArguments(x) =>
        Some(Res.Failure(
          None,
          Util.normalizeNewlines(
            s"""Too many args were passed to this script: ${x.map(literalize(_)).mkString(", ")}
                |expected arguments: $expectedMsg""".stripMargin
          )

        ))
      case Router.Result.Error.RedundantArguments(x) =>
        Some(Res.Failure(
          None,
          Util.normalizeNewlines(
            s"""Redundant values were passed for arguments: ${x.map(literalize(_)).mkString(", ")}
                |expected arguments: $expectedMsg""".stripMargin
          )
        ))
      case Router.Result.Error.InvalidArguments(x) =>
        Some(Res.Failure(
          None,
          "The following arguments failed to be parsed:" + Util.newLine +
            x.map{
              case Router.Result.ParamError.Missing(p) =>
                s"(${renderArg(p)}) was missing"
              case Router.Result.ParamError.Invalid(p, v, ex) =>
                s"(${renderArg(p)}) failed to parse input ${literalize(v)} with $ex"
              case Router.Result.ParamError.DefaultFailed(p, ex) =>
                s"(${renderArg(p)})'s default value failed to evaluate with $ex"
            }.mkString(Util.newLine) + Util.newLine + s"expected arguments: $expectedMsg"
        ))
    }
  }

  def renderArg(arg: ArgSig) = backtickWrap(arg.name) + ": " + arg.typeString


  def mainMethodDetails(ep: EntryPoint) = {
    ep.argSignatures.collect{
      case ArgSig(name, tpe, Some(doc), default) =>
        Util.newLine + name + " // " + doc
    }.mkString
  }

  /**
    * Additional [[scopt.Read]] instance to teach it how to read Ammonite paths
    */
  implicit def pathScoptRead: scopt.Read[Path] = scopt.Read.stringRead.map(Path(_, pwd))

}
