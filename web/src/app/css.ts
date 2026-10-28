// Each module brings its own stylesheet: loadCSS(new URL("./mine.css", import.meta.url)). The build copies src/**/*.css into dist.
const loaded = new Set<string>();

export function loadCSS(url: URL | string): Promise<void> {
  const href = String(url);
  if (loaded.has(href)) return Promise.resolve();
  loaded.add(href);
  return new Promise((resolve) => {
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = href;
    l.onload = l.onerror = () => resolve();
    document.head.append(l);
  });
}
