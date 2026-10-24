# personal website

# dev

to run 
```
bundle exec jekyll serve --livereload
```

If embedded videos stop immediately in Chrome on Jekyll's local server,
keep Jekyll running to rebuild changes and serve the generated site from a
second terminal:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory _site
```

Preview at http://localhost:4173 and refresh after edits. This uses the same
generated files while avoiding the playback issue on Jekyll's server.
