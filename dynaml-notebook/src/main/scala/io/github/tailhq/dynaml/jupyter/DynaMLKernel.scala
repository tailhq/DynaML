/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.tailhq.dynaml.jupyter

import almond.util.ThreadUtil.singleThreadedExecutionContext
import almond.channels.zeromq.ZeromqThreads
import almond.kernel.install.Install
import almond.kernel.{Kernel, KernelThreads}
import almond.logger.{Level, LoggerContext}
import caseapp._


object DynaMLKernel extends CaseApp[Options] {

  def run(options: Options, args: RemainingArgs): Unit = {

    if (options.install)
      Install.installOrError(
        defaultId = "dynaml-scala",
        defaultDisplayName = "Dynaml/Scala",
        language = "scala",
        options = options.installOptions
      ) match {
        case Left(e) =>
          Console.err.println(s"Error: $e")
          sys.exit(1)
        case Right(dir) =>
          println(s"Installed DynaML Jupyter kernel under $dir")
          sys.exit(0)
      }

    val connectionFile = options.connectionFile.getOrElse {
      Console.err.println(
        "No connection file passed, and installation not asked. Run with --install to install the kernel, " +
          "or pass a connection file via --connection-file to run the kernel."
      )
      sys.exit(1)
    }

    val logCtx = Level.fromString(options.log) match {
      case Left(err) =>
        Console.err.println(err)
        sys.exit(1)
      case Right(level) =>
        LoggerContext.stderr(level)
    }

    val log = logCtx(getClass)

    val zeromqThreads = ZeromqThreads.create("dynaml-scala-kernel")
    val kernelThreads = KernelThreads.create("dynaml-scala-kernel")
    val interpreterEc = singleThreadedExecutionContext("dynaml-scala-interpreter")

    log.info("Running kernel")
    Kernel.create(new DynaMLJupyter(), interpreterEc, kernelThreads, logCtx)
      .flatMap(_.runOnConnectionFile(connectionFile, "dynaml", zeromqThreads))
      .unsafeRunSync()
  }
}