# Preserve Markdown before Jekyll converts it to HTML, resolving Liquid asset URLs.
Jekyll::Hooks.register :posts, :pre_render do |post, payload|
  next unless post.data["layout"] == "post"

  markdown = post.content
  if post.render_with_liquid?
    markdown = post.renderer.render_liquid(
      markdown, payload,
      { :registers => { :site => post.site, :page => payload["page"] } },
      post.path
    )
  end
  post.data["markdown_content"] = "# #{post.data['title']}\n\n#{markdown.strip}\n"
end
