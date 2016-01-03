require 'nokogiri'

module Jekyll

  module TOCGenerator

    TOGGLE_HTML = '<div id="toctitle"><h2>%1</h2>%2</div>'
    TOC_CONTAINER_HTML = '<div id="toc-container"><table class="toc" id="toc"><tbody><tr><td>%1<ul>%2</ul></td></tr></tbody></table></div>'
    HIDE_HTML = '<span class="toctoggle">[<a id="toctogglelink" class="internal" href="#">%1</a>]</span>'

    def toc_generate(html)
      # No Toc can be specified on every single page
      # For example the index page has no table of contents
      no_toc = @context.environments.first["page"]["noToc"] || false;

      return html if no_toc

      config = @context.registers[:site].config

      # Minimum number of items needed to show TOC, default 0 (0 means no minimum)
      min_items_to_show_toc = config["minItemsToShowToc"] || 0

      anchor_prefix = config["anchorPrefix"] || 'tocAnchor-'

      # better for traditional page seo, commonlly use h1 as title
      toc_top_tag = config["tocTopTag"] || 'h1'
      toc_top_tag = toc_top_tag.gsub(/h/, '').to_i

      toc_top_tag = 5 if toc_top_tag > 5

      toc_sec_tag = toc_top_tag + 1
      toc_top_tag = "h#{toc_top_tag}"
      toc_sec_tag = "h#{toc_sec_tag}"


      # Text labels
      contents_label     = config["contentsLabel"] || 'Contents'
      hide_label         = config["hideLabel"] || 'hide'
      # show_label       = config["showLabel"] || 'show' # unused
      show_toggle_button = config["showToggleButton"]

      toc_html = ''
      toc_level = 1
      toc_section = 1
      item_number = 1
      level_html = ''

      doc = Nokogiri::HTML(html)

      # Find H1 tag and all its H2 siblings until next H1
      doc.css(toc_top_tag).each do |tag|
        # TODO This XPATH expression can greatly improved
        ct    = tag.xpath("count(following-sibling::#{toc_top_tag})")
        sects = tag.xpath("following-sibling::#{toc_sec_tag}[count(following-sibling::#{toc_top_tag})=#{ct}]")

        level_html    = '';
        inner_section = 0;

        sects.map.each do |sect|
          inner_section += 1;
          anchor_id = [
                        anchor_prefix, toc_level, '-', toc_section, '-',
                        inner_section
                      ].map(&:to_s).join ''

          sect['id'] = "#{anchor_id}"

          level_html += create_level_html(anchor_id,
                                          toc_level + 1,
                                          toc_section + inner_section,
                                          item_number.to_s + '.' + inner_section.to_s,
                                          sect.text,
                                          '')
        end

        level_html = '<ul>' + level_html + '</ul>' if level_html.length > 0

        anchor_id = anchor_prefix + toc_level.to_s + '-' + toc_section.to_s;
        tag['id'] = "#{anchor_id}"

        toc_html += create_level_html(anchor_id,
                                      toc_level,
                                      toc_section,
                                      item_number,
                                      tag.text,
                                      level_html);

        toc_section += 1 + inner_section;
        item_number += 1;
      end

      # for convenience item_number starts from 1
      # so we decrement it to obtain the index count
      toc_index_count = item_number - 1

      return html unless toc_html.length > 0

      hide_html = '';
      hide_html = HIDE_HTML.gsub('%1', hide_label) if (show_toggle_button)

      if min_items_to_show_toc <= toc_index_count
        replaced_toggle_html = TOGGLE_HTML
        .gsub('%1', contents_label)
        .gsub('%2', hide_html);

        toc_table = TOC_CONTAINER_HTML
        .gsub('%1', replaced_toggle_html)
        .gsub('%2', toc_html);

        doc.css('body').children.before(toc_table)
      end

      doc.to_xhtml
    end

    private

    def create_level_html(anchor_id, toc_level, toc_section, tocNumber, tocText, tocInner)
      link = '<a href="#%1"><span class="tocnumber">%2</span> <span class="toctext">%3</span></a>%4'
      .gsub('%1', anchor_id.to_s)
      .gsub('%2', tocNumber.to_s)
      .gsub('%3', tocText)
      .gsub('%4', tocInner ? tocInner : '');
      '<li class="toc_level-%1 toc_section-%2">%3</li>'
      .gsub('%1', toc_level.to_s)
      .gsub('%2', toc_section.to_s)
      .gsub('%3', link)
    end

  end

end

Liquid::Template.register_filter(Jekyll::TOCGenerator)
