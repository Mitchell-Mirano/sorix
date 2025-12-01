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

// Forzar renderizado tras cada cambio de página (Importante para Material)
document$.subscribe(() => {
  MathJax.typesetPromise();
});
