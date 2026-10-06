"""Build the static website from the executed clustering and PCA notebooks.

Run from anywhere:  py web/build.py
Output: web/index.html, web/clustering.html, web/pca.html, web/images/*.png
        Retail_Sales_Insights.html in the project folder (single self-contained file)
"""

import base64
import html
import json
import re
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
IMG_DIR = WEB_DIR / "images"

PROJECT_TITLE = "Unit 3 Mini Project"
PROJECT_SUBTITLE = "Retail Sales Insights Using Clustering and PCA"

PAGES = [
    {
        "notebook": "Task5_Clustering.ipynb",
        "out": "clustering.html",
        "key": "clustering",
        "nav": "Clustering",
        "summary": "Group orders by quantity and unit price with K-Means, Gaussian Mixture "
                   "and Hierarchical clustering, then compare them using Silhouette and ARI.",
        "hide": ["Graph: Elbow method and Silhouette scores"],
    },
    {
        "notebook": "Task6_PCA.ipynb",
        "out": "pca.html",
        "key": "pca",
        "nav": "PCA",
        "summary": "Reduce the sales and city features with PCA, built from scratch and "
                   "verified with scikit-learn, and explore variance, loadings and a 2D view.",
    },
]


def strip_task_prefix(text):
    return re.sub(r"^\s*Task\s*\d+\s*[:\-]\s*", "", text, flags=re.I)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def inline_md(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def markdown_to_html(src, anchor_prefix=""):
    out, para, list_tag = [], [], None

    def flush_para():
        if para:
            out.append(f"<p>{inline_md(' '.join(para))}</p>")
            para.clear()

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for line in src.splitlines():
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        ordered = re.match(r"^\d+\.\s+(.*)", stripped)
        bullet = re.match(r"^[-*]\s+(.*)", stripped)

        if not stripped:
            flush_para()
            close_list()
        elif heading:
            flush_para()
            close_list()
            level = len(heading.group(1))
            text = strip_task_prefix(heading.group(2))
            anchor = f' id="{anchor_prefix}{slugify(text)}"' if level == 2 else ""
            out.append(f"<h{level}{anchor}>{inline_md(text)}</h{level}>")
        elif ordered or bullet:
            flush_para()
            tag = "ol" if ordered else "ul"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{inline_md((ordered or bullet).group(1))}</li>")
        else:
            close_list()
            para.append(stripped)

    flush_para()
    close_list()
    return "\n".join(out)


def clean_dataframe_html(raw):
    raw = re.sub(r"<style scoped>.*?</style>", "", raw, flags=re.S)
    return f'<div class="table-wrap">{raw.strip()}</div>'


def render_outputs(outputs, prefix, counter):
    parts, stream_buf = [], []

    def flush_stream():
        if stream_buf:
            text = "".join(stream_buf).strip("\n")
            parts.append(f'<pre class="output">{html.escape(text)}</pre>')
            stream_buf.clear()

    for out in outputs:
        kind = out["output_type"]
        if kind == "stream":
            stream_buf.append("".join(out["text"]))
            continue
        flush_stream()
        data = out.get("data", {})
        if "image/png" in data:
            counter[0] += 1
            name = f"{prefix}_fig{counter[0]}.png"
            png = data["image/png"]
            png = "".join(png) if isinstance(png, list) else png
            (IMG_DIR / name).write_bytes(base64.b64decode(png))
            parts.append(f'<figure><img src="images/{name}" alt="Figure {counter[0]}" loading="lazy"></figure>')
        elif "text/html" in data:
            parts.append(clean_dataframe_html("".join(data["text/html"])))
        elif "text/plain" in data:
            parts.append(f'<pre class="output">{html.escape("".join(data["text/plain"]))}</pre>')
        elif kind == "error":
            parts.append(f'<pre class="output error">{html.escape(out.get("ename", "Error"))}: '
                         f'{html.escape(out.get("evalue", ""))}</pre>')
    flush_stream()
    return "\n".join(parts)


STANDALONE_FILE = ROOT / "Retail_Sales_Insights.html"

STANDALONE_SCRIPT = """<script>
(function () {
  var views = document.querySelectorAll(".view");
  var links = document.querySelectorAll(".site-header nav a");
  function show() {
    var hash = location.hash.slice(1) || "home";
    var target = document.getElementById(hash);
    var view = target ? target.closest(".view") : null;
    if (!view) { view = document.getElementById("home"); target = null; }
    views.forEach(function (v) { v.hidden = v !== view; });
    links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + view.id); });
    if (target && target !== view) { target.scrollIntoView(); } else { window.scrollTo(0, 0); }
  }
  window.addEventListener("hashchange", show);
  window.addEventListener("load", show);
  show();
})();
</script>"""


def href_for(key, standalone):
    if standalone:
        return f"#{key}"
    return "index.html" if key == "home" else next(p["out"] for p in PAGES if p["key"] == key)


def header(active, standalone=False):
    links = [("home", "Home")] + [(p["key"], p["nav"]) for p in PAGES]
    items = "".join(
        f'<a href="{href_for(key, standalone)}"{' class="active"' if key == active else ""}>{label}</a>'
        for key, label in links
    )
    return (f'<header class="site-header"><div class="inner">'
            f'<a class="brand" href="{href_for("home", standalone)}">{PROJECT_TITLE}</a>'
            f"<nav>{items}</nav></div></header>")


def page(title, active, body):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} | {PROJECT_TITLE}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
{header(active)}
<main>
{body}
</main>
<footer class="site-footer">{PROJECT_TITLE} &middot; {PROJECT_SUBTITLE}</footer>
</body>
</html>
"""


def embed_images(body):
    def to_data_uri(match):
        data = base64.b64encode((IMG_DIR / match.group(1)).read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    body = body.replace(' loading="lazy"', "")
    return re.sub(r'src="images/([^"]+)"', to_data_uri, body)


def build_standalone(views):
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    sections = "\n".join(
        f'<div class="view" id="{key}">\n{embed_images(body)}\n</div>' for key, body in views
    )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PROJECT_SUBTITLE} | {PROJECT_TITLE}</title>
<style>
{css}
</style>
</head>
<body>
{header("home", standalone=True)}
<main>
{sections}
</main>
<footer class="site-footer">{PROJECT_TITLE} &middot; {PROJECT_SUBTITLE}</footer>
{STANDALONE_SCRIPT}
</body>
</html>
"""
    STANDALONE_FILE.write_text(doc, encoding="utf-8")


def build_notebook(cfg):
    nb = json.loads((ROOT / cfg["notebook"]).read_text(encoding="utf-8"))
    cells = nb["cells"]
    counter = [0]

    intro_src = "".join(cells[0]["source"])
    title_match = re.search(r"^#\s+(.*)", intro_src, flags=re.M)
    title = strip_task_prefix(title_match.group(1)) if title_match else cfg["nav"]
    intro_html = markdown_to_html(re.sub(r"^#\s+.*\n?", "", intro_src, count=1, flags=re.M))

    sections, toc, current = [], [], []
    anchor_prefix = f"{cfg['key']}-"
    hidden = set(cfg.get("hide", []))
    skip_next_code = False

    def close_section():
        if current:
            sections.append('<section class="card">' + "\n".join(current) + "</section>")
            current.clear()

    for cell in cells[1:]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and re.sub(r"^#+\s*", "", source.strip()) in hidden:
            skip_next_code = True
            continue
        if cell["cell_type"] == "code" and skip_next_code:
            skip_next_code = False
            continue
        if cell["cell_type"] == "markdown":
            h2 = re.match(r"^##\s+(.*)", source.strip())
            if h2:
                close_section()
                text = strip_task_prefix(h2.group(1))
                toc.append((f"{anchor_prefix}{slugify(text)}", re.sub(r"^\d+\.\s*", "", text)))
            current.append(markdown_to_html(source, anchor_prefix))
        elif cell["cell_type"] == "code" and source.strip():
            current.append(
                '<details class="code"><summary>Show code</summary>'
                f"<pre><code>{html.escape(source)}</code></pre></details>"
            )
            outputs = render_outputs(cell.get("outputs", []), cfg["key"], counter)
            if outputs:
                current.append(outputs)
    close_section()

    toc_html = "".join(f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor, label in toc)
    body = f"""<section class="page-hero">
<p class="eyebrow">{PROJECT_SUBTITLE}</p>
<h1>{html.escape(title)}</h1>
<div class="intro">{intro_html}</div>
<nav class="toc">{toc_html}</nav>
</section>
{"".join(sections)}"""

    (WEB_DIR / cfg["out"]).write_text(page(title, cfg["key"], body), encoding="utf-8")
    return title, body


def index_body(titles, standalone=False):
    cards = "\n".join(
        f"""<a class="project-card" href="{href_for(cfg['key'], standalone)}">
<div class="project-card-body">
<h2>{html.escape(title)}</h2>
<p>{html.escape(cfg['summary'])}</p>
<span class="more">View analysis &rarr;</span>
</div>
</a>"""
        for cfg, title in zip(PAGES, titles)
    )
    body = f"""<section class="home-hero">
<p class="subtitle">{PROJECT_TITLE}</p>
<h1>{PROJECT_SUBTITLE}</h1>
<p class="meta">Dataset: <code>store_sales_dataset.csv</code> &middot; 500 orders</p>
</section>
<div class="project-grid">
{cards}
</div>"""
    return body


def main():
    IMG_DIR.mkdir(exist_ok=True)
    for old in IMG_DIR.glob("*.png"):
        old.unlink()
    for stale in ("task5.html", "task6.html"):
        (WEB_DIR / stale).unlink(missing_ok=True)
    built = [build_notebook(cfg) for cfg in PAGES]
    titles = [title for title, _ in built]
    (WEB_DIR / "index.html").write_text(page("Home", "home", index_body(titles)), encoding="utf-8")
    build_standalone(
        [("home", index_body(titles, standalone=True))]
        + [(cfg["key"], body) for cfg, (_, body) in zip(PAGES, built)]
    )
    print("Built:", ", ".join(["index.html"] + [cfg["out"] for cfg in PAGES]))
    print("Standalone:", STANDALONE_FILE)


if __name__ == "__main__":
    main()
