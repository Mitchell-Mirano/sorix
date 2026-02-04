window.MathJax = {
  tex: {
    // Soporta ambas sintaxis: notebooks y MkDocs
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Procesar explícitamente arithmatex y contenido de notebooks
    ignoreHtmlClass: 'mathjax_ignore',
    processHtmlClass: 'arithmatex|output_area'
  }
};

document$.subscribe(() => {
  if (window.MathJax) {
    setTimeout(() => {
      MathJax.typesetClear();
      MathJax.typesetPromise();
    }, 50);
  }
});
