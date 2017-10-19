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
package io.github.mandar2812.dynaml

import java.io.{InputStream, OutputStream, PrintStream}

import ammonite.ops.Path
import ammonite.runtime.Storage
import ammonite.sshd.{SshServer, SshServerConfig}
import ammonite.sshd.util.Environment
import ammonite.util.{Bind, Colors}


/**
  * An ssh server which serves ammonite repl as it's shell channel.
  * To start listening for incoming connections call
  * [[start()]] method. You can [[stop()]] the server at any moment.
  * It will also close all running sessions
  * @param sshConfig configuration of ssh server,
  *                  such as users credentials or port to be listening to
  * @param predef predef that will be installed on repl instances served by this server
  * @param replArgs arguments to pass to ammonite repl on initialization of the session;
  *                 an argument named "session" containing the SSHD session will be added
  * @param classLoader classloader for ammonite to use
  */
class DynaMLSSH(
  sshConfig: SshServerConfig,
  predef: String = "",
  defaultPredef: Boolean = true,
  wd: Path = ammonite.ops.pwd,
  replArgs: Seq[Bind[_]] = Nil,
  classLoader: ClassLoader = DynaMLSSH.getClass.getClassLoader) {
  private lazy val sshd = SshServer(
    sshConfig,
    shellServer = DynaMLSSH.runRepl(
      sshConfig.ammoniteHome,
      predef,
      defaultPredef,
      wd,
      replArgs,
      classLoader
    )
  )

  def port = sshd.getPort
  def start(): Unit = sshd.start()
  def stop(): Unit = sshd.stop()
  def stopImmediately(): Unit = sshd.stop(true)
}


object DynaMLSSH {
  // Actually runs a repl inside of session serving a remote user shell.
  private def runRepl(
    homePath: Path,
    predefCode: String,
    defaultPredef: Boolean,
    wd: Path,
    replArgs: Seq[Bind[_]],
    replServerClassLoader: ClassLoader)(
    in: InputStream, out: OutputStream): Unit = {
    // since sshd server has it's own customised environment,
    // where things like System.out will output to the
    // server's console, we need to prepare individual environment
    // to serve this particular user's session

    Environment.withEnvironment(Environment(replServerClassLoader, in, out)) {
      try {
        DynaML(
          predefCode = predefCode,
          predefFile = None,
          defaultPredef = defaultPredef,
          storageBackend = new Storage.Folder(homePath),
          wd = wd,
          inputStream = in, outputStream = out, errorStream = out,
          verboseOutput = false,
          remoteLogging = false,
          colors = Colors.Default
        ).run(replArgs:_*)
      } catch {
        case any: Throwable =>
          val sshClientOutput = new PrintStream(out)
          sshClientOutput.println("What a terrible failure, DynaML just blew up!")
          any.printStackTrace(sshClientOutput)
      }
    }
  }
}
