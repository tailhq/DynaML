package io.github.tailhq.dynaml.utils.sumac.examples

import io.github.tailhq.dynaml.utils.sumac.{FieldArgs, ArgMain}

/**
 * after you compile the library, run this with
 *
 * java -cp core/target/scala-2.9.3/classes/:$SCALA_HOME/lib/scala-library.jar com.quantifind.sumac.examples.SimpleApp <args>
 *
 * eg.
 *
 * java -cp core/target/scala-2.9.3/classes/:$SCALA_HOME/lib/scala-library.jar com.quantifind.sumac.examples.SimpleApp
 * or
 * java -cp core/target/scala-2.9.3/classes/:$SCALA_HOME/lib/scala-library.jar com.quantifind.sumac.examples.SimpleApp --count 2
 * or
 * java -cp core/target/scala-2.9.3/classes/:$SCALA_HOME/lib/scala-library.jar com.quantifind.sumac.examples.SimpleApp --name ooga
 * or
 * java -cp core/target/scala-2.9.3/classes/:$SCALA_HOME/lib/scala-library.jar com.quantifind.sumac.examples.SimpleApp --help
 *
 * etc.
 *
 */
object SimpleApp extends ArgMain[SimpleAppArgs] {
  def main(args: SimpleAppArgs) {
    (0 until args.count).foreach{_ => println(args.name)}
  }
}

class SimpleAppArgs extends FieldArgs {
  var name: String = "the default name"
  var count: Int = 5
}
