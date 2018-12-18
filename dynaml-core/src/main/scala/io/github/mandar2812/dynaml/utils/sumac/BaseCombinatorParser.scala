package io.github.mandar2812.dynaml.utils.sumac

import scala.util.parsing.combinator.RegexParsers

/**
 * The base combinator parser definitions to parse a list of items, where items can be quoted and include the list
 *   separator or not.
 * User: andrews
 * Date: 3/28/14
 */
trait BaseCombinatorParser[T] extends RegexParsers {

  /**
   * extra characters to forbid in unquoted tokens
   */
  protected def extraForbiddenChars: String

  /**
   * the separator in the list
   */
  protected val separator: String = ","

  /**
   * what makes an item in the list
   */
  protected val item: Parser[T]

  //need to be lazy as we are waiting for extraFrobiddenChars to exist
  //either it's not quoted, there might be at most one single or double quote in it (simplification)
  private lazy val noQuote = s"""[^"'$extraForbiddenChars$separator]*["']?[^"'$extraForbiddenChars$separator]*""".r
  //or it's quoted with single or double quotes and anything goes except the quote
  private lazy val quoted = "\"" ~> """[^"]+""".r <~ "\"" | "'" ~> "[^']+".r <~ "'"

  /**
   * a token is either anything without a separator in it, or a quoted string that might contain the separator
   */
  protected lazy val token: Parser[String] = quoted | noQuote

  /**
   * a list of items
   */
  protected val list: Parser[Seq[T]] = rep1(item, separator ~> item)

  /**
   * parse the list of items
   * @param in
   * @return
   */
  def parse(in: String): Seq[T] = parseAll(list, in) match {
    case Success(result, _) => result
    case failure: NoSuccess =>
      throw new IllegalArgumentException(s"'$in' cannot be parsed. Caused by: ${failure.msg}}")
  }

}
