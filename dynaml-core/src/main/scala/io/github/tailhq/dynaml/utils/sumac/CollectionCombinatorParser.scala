package io.github.tailhq.dynaml.utils.sumac

/**
 * Parse a list separated by commas, allow items in the list to be quoted so they can include commas.
 * User: pierre
 * Date: 3/25/14
 */
object CollectionCombinatorParser  extends BaseCombinatorParser[String] {


  override val extraForbiddenChars: String = ""
  override val item = token

  def apply(in: String): Seq[String] = parse(in)

}
