package io.github.mandar2812.dynaml.repl

import ammonite.runtime.Evaluator.{evaluatorRunPrinter, userCodeExceptionHandler}
import ammonite.runtime.{Evaluator, Frame}
import ammonite.util.Util.ClassFiles
import ammonite.util._

import scala.util.Try

abstract class DynaMLEvaluator extends Evaluator {
  def processCell(classFiles: Util.ClassFiles,
                  newImports: Imports,
                  printer: Printer,
                  indexedWrapperName: Name,
                  silent: Boolean,
                  contextClassLoader: ClassLoader): Res[(Iterator[String], Evaluated)]
}

object DynaMLEvaluator {
  def apply(headFrame: => Frame): DynaMLEvaluator = new DynaMLEvaluator { eval =>


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
        cls.getDeclaredMethod("$main").invoke(null)
      }

    def processLine(classFiles: Util.ClassFiles,
                    newImports: Imports,
                    printer: Printer,
                    indexedWrapperName: Name,
                    silent: Boolean,
                    contextClassLoader: ClassLoader) = {
      for {
        cls <- loadClass("ammonite.$sess." + indexedWrapperName.backticked, classFiles)
        _ <- Catching{userCodeExceptionHandler}
      } yield {
        // Exhaust the printer iterator now, before exiting the `Catching`
        // block, so any exceptions thrown get properly caught and handled
        val iter = evalMain(cls, contextClassLoader).asInstanceOf[Iterator[String]]

        if (!silent) evaluatorRunPrinter(iter.foreach(printer.resultStream.print))
        else evaluatorRunPrinter(iter.foreach(_ => ()))

        // "" Empty string as cache tag of repl code
        evaluationResult(Seq(Name("ammonite"), Name("$sess"), indexedWrapperName), newImports)
      }
    }

    def processCell(classFiles: Util.ClassFiles,
                    newImports: Imports,
                    printer: Printer,
                    indexedWrapperName: Name,
                    silent: Boolean,
                    contextClassLoader: ClassLoader) = {
      for {
        cls <- loadClass("ammonite.$sess." + indexedWrapperName.backticked, classFiles)
        _ <- Catching{userCodeExceptionHandler}
      } yield {
        // Exhaust the printer iterator now, before exiting the `Catching`
        // block, so any exceptions thrown get properly caught and handled
        val iter = evalMain(cls, contextClassLoader).asInstanceOf[Iterator[String]]

        if (!silent) evaluatorRunPrinter(iter.foreach(printer.resultStream.print))
        else evaluatorRunPrinter(iter.foreach(_ => ()))

        // "" Empty string as cache tag of repl code
        (iter, evaluationResult(Seq(Name("ammonite"), Name("$sess"), indexedWrapperName), newImports))
      }
    }



    def processScriptBlock(cls: Class[_],
                           newImports: Imports,
                           wrapperName: Name,
                           pkgName: Seq[Name],
                           contextClassLoader: ClassLoader) = {
      for {
        _ <- Catching{userCodeExceptionHandler}
      } yield {
        evalMain(cls, contextClassLoader)
        val res = evaluationResult(pkgName :+ wrapperName, newImports)
        res
      }
    }

    def evaluationResult(wrapperName: Seq[Name],
                         imports: Imports) = {
      Evaluated(
        wrapperName,
        Imports(
          for(id <- imports.value) yield {
            val filledPrefix =
              if (id.prefix.isEmpty) {
                // For some reason, for things not-in-packages you can't access
                // them off of `_root_`
                wrapperName
              } else {
                id.prefix
              }
            val rootedPrefix: Seq[Name] =
              if (filledPrefix.headOption.exists(_.backticked == "_root_")) filledPrefix
              else Seq(Name("_root_")) ++ filledPrefix

            id.copy(prefix = rootedPrefix)
          }
        )
      )
    }
  }

}