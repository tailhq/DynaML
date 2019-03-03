package io.github.mandar2812.dynaml.repl

import java.nio.charset.Charset

import ammonite.runtime.Evaluator.{evaluatorRunPrinter, userCodeExceptionHandler}
import ammonite.runtime.{Evaluator, Frame}
import ammonite.util.Util.ClassFiles
import ammonite.util._
import org.apache.commons.io.output.ByteArrayOutputStream

import scala.util.Try

abstract class DynaMLEvaluator extends Evaluator {
  def processCell(classFiles: ClassFiles,
                  newImports: Imports,
                  usedEarlierDefinitions: Seq[String],
                  printer: Printer,
                  indexedWrapperName: Name,
                  wrapperPath: Seq[Name],
                  silent: Boolean,
                  contextClassLoader: ClassLoader): Res[Evaluated]
}

object DynaMLEvaluator {
  def apply(headFrame: => Frame): DynaMLEvaluator = new DynaMLEvaluator { eval =>


    def processCell(classFiles: ClassFiles,
                    newImports: Imports,
                    usedEarlierDefinitions: Seq[String],
                    printer: Printer,
                    indexedWrapperName: Name,
                    wrapperPath: Seq[Name],
                    silent: Boolean,
                    contextClassLoader: ClassLoader) = {
      for {
        cls <- loadClass("ammonite.$sess." + indexedWrapperName.backticked, classFiles)
        _ <- Catching{userCodeExceptionHandler}
      } yield {
        headFrame.usedEarlierDefinitions = usedEarlierDefinitions

        // Exhaust the printer iterator now, before exiting the `Catching`
        // block, so any exceptions thrown get properly caught and handled
        val iter = evalMain(cls, contextClassLoader).asInstanceOf[Iterator[String]]

        if (!silent) evaluatorRunPrinter(iter.foreach(printer.resultStream.print))
        else evaluatorRunPrinter(iter.foreach(_ => ()))

        // "" Empty string as cache tag of repl code
        evaluationResult(
          Seq(Name("ammonite"), Name("$sess"), indexedWrapperName),
          wrapperPath,
          newImports
        )
      }
    }



    def loadClass(fullName: String, classFiles: ClassFiles): Res[Class[_]] = {
      Res[Class[_]](
        Try {
          for ((name, bytes) <- classFiles.sortBy(_._1)) {
            headFrame.classloader.addClassFile(name, bytes)
          }

          headFrame.classloader.findClass(fullName)
        },
        e =>"Failed to load compiled class " + e
      )
    }


    def evalMain(cls: Class[_], contextClassloader: ClassLoader) =
      Util.withContextClassloader(contextClassloader){

        val (method, instance) =
          try {
            (cls.getDeclaredMethod("$main"), null)
          } catch {
            case e: NoSuchMethodException =>
              // Wrapper with very long names seem to require this
              try {
                val cls0 = contextClassloader.loadClass(cls.getName + "$")
                val inst = cls0.getDeclaredField("MODULE$").get(null)
                (cls0.getDeclaredMethod("$main"), inst)
              } catch {
                case _: ClassNotFoundException | _: NoSuchMethodException =>
                  throw e
              }
          }

        method.invoke(instance)
      }

    def processLine(classFiles: Util.ClassFiles,
                    newImports: Imports,
                    usedEarlierDefinitions: Seq[String],
                    printer: Printer,
                    indexedWrapperName: Name,
                    wrapperPath: Seq[Name],
                    silent: Boolean,
                    contextClassLoader: ClassLoader) = {
      for {
        cls <- loadClass("ammonite.$sess." + indexedWrapperName.backticked, classFiles)
        _ <- Catching{userCodeExceptionHandler}
      } yield {
        headFrame.usedEarlierDefinitions = usedEarlierDefinitions

        // Exhaust the printer iterator now, before exiting the `Catching`
        // block, so any exceptions thrown get properly caught and handled
        val iter = evalMain(cls, contextClassLoader).asInstanceOf[Iterator[String]]

        if (!silent) evaluatorRunPrinter(iter.foreach(printer.resultStream.print))
        else evaluatorRunPrinter(iter.foreach(_ => ()))

        // "" Empty string as cache tag of repl code
        evaluationResult(
          Seq(Name("ammonite"), Name("$sess"), indexedWrapperName),
          wrapperPath,
          newImports
        )
      }
    }


    def processScriptBlock(cls: Class[_],
                           newImports: Imports,
                           usedEarlierDefinitions: Seq[String],
                           wrapperName: Name,
                           wrapperPath: Seq[Name],
                           pkgName: Seq[Name],
                           contextClassLoader: ClassLoader) = {
      for {
        _ <- Catching{userCodeExceptionHandler}
      } yield {
        headFrame.usedEarlierDefinitions = usedEarlierDefinitions
        evalMain(cls, contextClassLoader)
        val res = evaluationResult(pkgName :+ wrapperName, wrapperPath, newImports)
        res
      }
    }

    def evaluationResult(wrapperName: Seq[Name],
                         internalWrapperPath: Seq[Name],
                         imports: Imports) = {
      Evaluated(
        wrapperName,
        Imports(
          for(id <- imports.value) yield {
            val filledPrefix =
              if (internalWrapperPath.isEmpty) {
                val filledPrefix =
                  if (id.prefix.isEmpty) {
                    // For some reason, for things not-in-packages you can't access
                    // them off of `_root_`
                    wrapperName
                  } else {
                    id.prefix
                  }

                if (filledPrefix.headOption.exists(_.backticked == "_root_")) filledPrefix
                else Seq(Name("_root_")) ++ filledPrefix
              } else if (id.prefix.isEmpty)
              // For some reason, for things not-in-packages you can't access
              // them off of `_root_`
                Seq(Name("_root_")) ++ wrapperName ++ internalWrapperPath
              else if (id.prefix.startsWith(wrapperName))
                Seq(Name("_root_")) ++ wrapperName.init ++
                  Seq(id.prefix.apply(wrapperName.length)) ++ internalWrapperPath ++
                  id.prefix.drop(wrapperName.length + 1)
              else if (id.prefix.headOption.exists(_.backticked == "_root_"))
                id.prefix
              else
                Seq(Name("_root_")) ++ id.prefix

            id.copy(prefix = filledPrefix)
          }
        )
      )
    }
  }

}