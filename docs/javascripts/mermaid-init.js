document$.subscribe(function() {
  mermaid.initialize({ startOnLoad: true });
  mermaid.contentLoaded();
});
