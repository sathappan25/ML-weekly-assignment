"""Build Retail_Sales_Insights.html: every task's code and output, grouped by unit.

Run from anywhere:  py web/build.py

Each TaskN.py script is split into sections at its section-header comments and
executed once. Printed text, DataFrames (as tables) and graphs are captured per
section and embedded, with the code, in one HTML file in the project folder, so
the file can be opened anywhere without a server.
"""

import base64
import csv
import html
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
OUTPUT_FILE = ROOT / "Retail_Sales_Insights.html"
DATASET_FILE = ROOT / "store_sales_dataset.csv"

SITE_TITLE = "Retail Sales Insights"
SITE_HEADING = "Machine Learning Practicals"

UNITS = [
    {
        "number": 1,
        "title": "Data Loading and Exploration",
        "summary": "Load the store sales dataset, view it and summarise it with descriptive statistics.",
        "tasks": [1, 2],
    },
    {
        "number": 2,
        "title": "Linear Models for Regression and Classification",
        "summary": "Predict sales with linear regression and classify product category with "
                   "Bayesian logistic regression and SVM.",
        "tasks": [3, 4],
    },
    {
        "number": 3,
        "title": "Retail Sales Insights Using Clustering and PCA",
        "summary": "Group orders with K-Means, Gaussian Mixture and Hierarchical clustering, "
                   "and reduce dimensions with PCA.",
        "tasks": [5, 6],
    },
    {
        "number": 4,
        "title": "Hidden Markov Models",
        "summary": "Model the order sequence with a hidden Markov model and predict the next order's sales level.",
        "tasks": [7],
    },
    {
        "number": 5,
        "title": "Combining Models",
        "summary": "Categorise orders with a CART decision tree and compare ensemble methods for classification.",
        "tasks": [8, 9],
    },
]

TASKS = {
    1: {
        "script": "Task1_Load_and_View.py",
        "title": "Load and View the Dataset",
        "summary": "Load the CSV with pandas and look at its rows, shape, columns and data types.",
    },
    2: {
        "script": "Task2_Summary_and_Statistics.py",
        "title": "Summary and Statistics",
        "summary": "Descriptive statistics, missing values, unique values and sales totals "
                   "by category, city and product.",
    },
    3: {
        "script": "Task3_Linear_Regression.py",
        "title": "Linear Regression",
        "summary": "Predict Total_Sales from Unit_Price, evaluate with MAE, MSE, RMSE and R\u00b2, and show the limits of a single feature.",
    },
    4: {
        "script": "Task4_Bayesian_LR_and_SVM.py",
        "title": "Bayesian Logistic Regression and SVM",
        "summary": "Classify product category with a PyMC Bayesian logistic regression and an RBF SVM.",
    },
    5: {
        "script": "Task5_Clustering.py",
        "title": "K-Means, Gaussian Mixture and Hierarchical Clustering",
        "summary": "Group orders by quantity and unit price and compare the three clustering methods.",
        "hide_figures": [1],
    },
    6: {
        "script": "Task6_PCA.py",
        "title": "Principal Component Analysis",
        "summary": "PCA from scratch and with scikit-learn: explained variance, loadings and a 2D projection.",
    },
    7: {
        "script": "Task7_HMM.py",
        "title": "Hidden Markov Model for Sequential Data",
        "summary": "Forward-backward, Baum-Welch and Viterbi from scratch to predict the next order's sales level.",
    },
    8: {
        "script": "Task8_CART.py",
        "title": "CART for Categorization",
        "summary": "Gini-based decision tree to categorise orders, with depth chosen by cross-validation.",
    },
    9: {
        "script": "Task9_Ensemble_Learning.py",
        "title": "Ensemble Learning for Classification",
        "summary": "Bagging, Random Forest, AdaBoost, Gradient Boosting and soft voting compared "
                   "with a single tree.",
    },
}

# Executed in a child process. Runs the instrumented task source; every plt.show()
# figure and every DataFrame.to_string() table is saved to out_dir and replaced in
# stdout by a marker, so it can be placed at the right point in the page.
RUNNER = r"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

script, source_path, out_dir, prefix = sys.argv[1:5]
counts = {"fig": 0, "table": 0}

def show(*args, **kwargs):
    for num in plt.get_fignums():
        counts["fig"] += 1
        name = f"{prefix}_fig{counts['fig']}.png"
        plt.figure(num).savefig(f"{out_dir}/{name}", dpi=110, bbox_inches="tight")
        print(f"[[FIGURE:{name}]]", flush=True)
    plt.close("all")

original_to_string = pd.DataFrame.to_string

def to_string(self, buf=None, *args, **kwargs):
    if buf is not None:
        return original_to_string(self, buf, *args, **kwargs)
    counts["table"] += 1
    name = f"{prefix}_table{counts['table']}.html"
    with open(f"{out_dir}/{name}", "w", encoding="utf-8") as f:
        f.write(self.to_html(index=kwargs.get("index", True), border=0))
    return f"[[TABLE:{name}]]"

plt.show = show
pd.DataFrame.to_string = to_string
sys.argv = [script]
with open(source_path, encoding="utf-8") as f:
    source = f.read()
exec(compile(source, script, "exec"), {"__name__": "__main__", "__file__": script})
"""

DASH_LINE = re.compile(r"^(\s*)# -{10,}\s*$")
TOKEN = re.compile(r"\[\[(FIGURE|TABLE|SECTION):([^\]]+)\]\]\n?")
CAPTION_LINE = re.compile(r"^Opening (?:graph|scatter plot)[^:]*:\s*(.*?)\.*$")

SCRIPT = """<script>
(function () {
  var views = document.querySelectorAll(".view");
  var links = document.querySelectorAll(".site-header nav a");
  function show() {
    var target = document.getElementById(location.hash.slice(1) || "home");
    var view = target ? target.closest(".view") : null;
    if (!view) { view = document.getElementById("home"); target = null; }
    views.forEach(function (v) { v.hidden = v !== view; });
    links.forEach(function (a) { a.classList.toggle("active", a.dataset.unit === view.dataset.unit); });
    setTimeout(function () {
      if (target && target !== view) { target.scrollIntoView(); } else { window.scrollTo(0, 0); }
    }, 0);
  }
  window.addEventListener("hashchange", show);
  window.addEventListener("load", show);
  show();

  document.querySelectorAll(".sheet tbody").forEach(function (body) {
    body.addEventListener("click", function (e) {
      var cell = e.target.closest("td");
      if (!cell) { return; }
      var old = body.querySelector("td.selected");
      if (old) { old.classList.remove("selected"); }
      cell.classList.add("selected");
    });
  });
})();
</script>"""


# --------------------------------------------------
# Splitting scripts into sections
# --------------------------------------------------
def find_headers(lines):
    """Section headers: dashed comment blocks, or (if none) top-level comments after a blank line."""
    headers = []
    if any(DASH_LINE.match(line) for line in lines):
        i = 0
        while i < len(lines):
            match = DASH_LINE.match(lines[i])
            if match:
                j, comment = i + 1, []
                while j < len(lines) and not DASH_LINE.match(lines[j]) and lines[j].strip().startswith("#"):
                    comment.append(lines[j].strip().lstrip("#").strip())
                    j += 1
                if j < len(lines) and DASH_LINE.match(lines[j]) and comment:
                    headers.append({"start": i, "body": j + 1, "indent": match.group(1),
                                    "title": comment[0], "note": comment[1:]})
                    i = j + 1
                    continue
            i += 1
    else:
        for i, line in enumerate(lines):
            if line.startswith("# ") and (i == 0 or not lines[i - 1].strip()):
                headers.append({"start": i, "body": i + 1, "indent": "",
                                "title": line[2:].strip(), "note": []})
    return headers


def split_script(source):
    """Return (sections, instrumented_source). Section 0 is everything before the first header."""
    lines = source.splitlines()
    headers = find_headers(lines)

    preamble = "\n".join(lines[: headers[0]["start"] if headers else len(lines)]).strip("\n")
    sections = [{
        "title": "Imports and helper functions" if re.search(r"^\s*def ", preamble, re.M) else "Import libraries",
        "note": [],
        "code": preamble,
    }]
    for k, h in enumerate(headers):
        end = headers[k + 1]["start"] if k + 1 < len(headers) else len(lines)
        sections.append({
            "title": re.sub(r"^\d+\.\s*", "", h["title"]),
            "note": h["note"],
            "code": "\n".join(lines[h["body"]:end]).strip("\n"),
        })

    instrumented = list(lines)
    for k in range(len(headers) - 1, -1, -1):
        h = headers[k]
        instrumented.insert(h["body"], f'{h["indent"]}print("[[SECTION:{k + 1}]]", flush=True)')
    return sections, "\n".join(instrumented) + "\n"


# --------------------------------------------------
# Running scripts and rendering their output
# --------------------------------------------------
def run_task(number, out_dir):
    cfg = TASKS[number]
    source = (ROOT / cfg["script"]).read_text(encoding="utf-8")
    sections, instrumented = split_script(source)
    source_path = out_dir / f"task{number}_instrumented.py"
    source_path.write_text(instrumented, encoding="utf-8")

    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, cfg["script"], str(source_path), str(out_dir), f"task{number}"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"{cfg['script']} failed:\n{result.stderr[-3000:]}")
    print(f"  ran {cfg['script']}")
    return sections, result.stdout


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def add_text(parts, text):
    """Append a text block; return a graph caption if the block ends with 'Opening graph: ...'."""
    lines = text.rstrip().splitlines()
    caption = None
    if lines and CAPTION_LINE.match(lines[-1].strip()):
        caption = CAPTION_LINE.match(lines[-1].strip()).group(1)
        lines = lines[:-1]
    block = "\n".join(lines).strip("\n")
    if block.strip():
        parts.append(f'<pre class="output">{html.escape(block)}</pre>')
    return caption


def split_output(stdout, n_sections, out_dir, hide_figures):
    outputs = [[] for _ in range(n_sections)]
    current, pos, caption = 0, 0, None
    for match in TOKEN.finditer(stdout):
        caption = add_text(outputs[current], stdout[pos:match.start()]) or caption
        pos = match.end()
        kind, value = match.groups()
        if kind == "SECTION":
            current, caption = int(value), None
        elif kind == "TABLE":
            table = (out_dir / value).read_text(encoding="utf-8")
            outputs[current].append(f'<div class="table-wrap">{table}</div>')
            caption = None
        else:
            fig_number = int(re.search(r"_fig(\d+)\.png$", value).group(1))
            if fig_number not in hide_figures:
                cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
                outputs[current].append(
                    f'<figure><img src="{data_uri(out_dir / value)}" alt="{html.escape(caption or value)}">{cap}</figure>'
                )
            caption = None
    add_text(outputs[current], stdout[pos:])
    return outputs


# --------------------------------------------------
# Page views
# --------------------------------------------------
def unit_of(task_number):
    return next(u for u in UNITS if task_number in u["tasks"])


def breadcrumb(*items):
    crumbs = [f'<a href="#{href}">{html.escape(label)}</a>' if href else f"<span>{html.escape(label)}</span>"
              for label, href in items]
    return f'<nav class="breadcrumb">{" / ".join(crumbs)}</nav>'


def card(href, title, summary, more):
    return f"""<a class="project-card" href="#{href}">
<h2>{html.escape(title)}</h2>
<p>{html.escape(summary)}</p>
<span class="more">{more} &rarr;</span>
</a>"""


def read_dataset():
    with DATASET_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def column_letter(index):
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def home_view(columns, rows):
    cards = "\n".join(
        card(f"unit-{u['number']}", f"Unit {u['number']}: {u['title']}", u["summary"], "View tasks")
        for u in UNITS
    )
    return f"""<section class="view" id="home" data-unit="home">
<section class="home-hero">
<p class="subtitle">{SITE_HEADING}</p>
<h1>{SITE_TITLE}</h1>
</section>
<a class="dataset-card" href="#dataset">
<div>
<p class="eyebrow">Dataset</p>
<h2>{DATASET_FILE.name}</h2>
<p>{len(rows)} rows &times; {len(columns)} columns: {html.escape(", ".join(columns))}</p>
</div>
<span class="more">View data &rarr;</span>
</a>
<div class="project-grid">
{cards}
</div>
</section>"""


def dataset_view(columns, rows):
    numeric = [all(is_number(r[i]) for r in rows) for i in range(len(columns))]
    letters = "".join(f"<th>{column_letter(i)}</th>" for i in range(len(columns)))
    names = "".join(f'<th class="{"num" if numeric[i] else ""}">{html.escape(c)}</th>' for i, c in enumerate(columns))
    body = "\n".join(
        f'<tr><th class="row-num">{n}</th>'
        + "".join(f'<td class="{"num" if numeric[i] else ""}">{html.escape(v)}</td>' for i, v in enumerate(r))
        + "</tr>"
        for n, r in enumerate(rows, start=2)
    )
    return f"""<section class="view" id="dataset" data-unit="dataset" hidden>
{breadcrumb(("Home", "home"), ("Dataset", None))}
<section class="page-hero">
<p class="eyebrow">Dataset</p>
<h1>{DATASET_FILE.name}</h1>
</section>
<div class="sheet-wrap">
<table class="sheet">
<thead>
<tr class="letters"><th class="corner"></th>{letters}</tr>
<tr class="names"><th class="row-num">1</th>{names}</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</div>
</section>"""


def unit_view(u):
    cards = "\n".join(
        card(f"task-{n}", f"Task {n}: {TASKS[n]['title']}", TASKS[n]["summary"], "View code and output")
        for n in u["tasks"]
    )
    return f"""<section class="view" id="unit-{u['number']}" data-unit="unit-{u['number']}" hidden>
{breadcrumb(("Home", "home"), (f"Unit {u['number']}", None))}
<section class="page-hero">
<p class="eyebrow">Unit {u['number']}</p>
<h1>{html.escape(u['title'])}</h1>
<div class="intro"><p>{html.escape(u['summary'])}</p></div>
</section>
<div class="project-grid">
{cards}
</div>
</section>"""


def task_view(number, sections, outputs, all_numbers):
    cfg = TASKS[number]
    u = unit_of(number)

    cards, toc, shown = [], [], 0
    for section, parts in zip(sections, outputs):
        if not section["code"].strip() and not parts:
            continue
        shown += 1
        anchor = f"task-{number}-s{shown}"
        title = f"{shown}. {section['title']}"
        toc.append(f'<a href="#{anchor}">{html.escape(section["title"])}</a>')
        note = (f'<p class="note">{"<br>".join(html.escape(n) for n in section["note"])}</p>'
                if section["note"] else "")
        code = (f'<details class="code"><summary>Show code</summary>'
                f'<pre><code>{html.escape(section["code"])}</code></pre></details>'
                if section["code"].strip() else "")
        cards.append(f'<section class="card" id="{anchor}"><h2>{html.escape(title)}</h2>'
                     f'{note}{code}{"".join(parts)}</section>')

    idx = all_numbers.index(number)
    prev_link = (f'<a href="#task-{all_numbers[idx - 1]}">&larr; Task {all_numbers[idx - 1]}</a>'
                 if idx > 0 else "<span></span>")
    next_link = (f'<a href="#task-{all_numbers[idx + 1]}">Task {all_numbers[idx + 1]} &rarr;</a>'
                 if idx < len(all_numbers) - 1 else "<span></span>")

    return f"""<section class="view" id="task-{number}" data-unit="unit-{u['number']}" hidden>
{breadcrumb(("Home", "home"), (f"Unit {u['number']}", f"unit-{u['number']}"), (f"Task {number}", None))}
<section class="page-hero">
<p class="eyebrow">Unit {u['number']} &middot; Task {number}</p>
<h1>{html.escape(cfg['title'])}</h1>
<div class="intro"><p>{html.escape(cfg['summary'])} Script: <code>{html.escape(cfg['script'])}</code></p></div>
<nav class="toc">{"".join(toc)}</nav>
</section>
{"".join(cards)}
<nav class="pager">{prev_link}{next_link}</nav>
</section>"""


def header():
    links = '<a href="#home" data-unit="home">Home</a>' + "".join(
        f'<a href="#unit-{u["number"]}" data-unit="unit-{u["number"]}">Unit {u["number"]}</a>' for u in UNITS
    ) + '<a href="#dataset" data-unit="dataset">Dataset</a>'
    return (f'<header class="site-header"><div class="inner">'
            f'<a class="brand" href="#home">{SITE_TITLE}</a><nav>{links}</nav></div></header>')


def main():
    numbers = [n for u in UNITS for n in u["tasks"]]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        print("Running task scripts...")
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = dict(zip(numbers, pool.map(lambda n: run_task(n, out_dir), numbers)))
        task_views = []
        for n in numbers:
            sections, stdout = runs[n]
            outputs = split_output(stdout, len(sections), out_dir, TASKS[n].get("hide_figures", []))
            task_views.append(task_view(n, sections, outputs, numbers))

    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    columns, rows = read_dataset()
    views = "\n".join(
        [home_view(columns, rows), dataset_view(columns, rows)] + [unit_view(u) for u in UNITS] + task_views
    )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{SITE_TITLE}</title>
<style>
{css}
</style>
</head>
<body>
{header()}
<main>
{views}
</main>
<footer class="site-footer">{SITE_TITLE} &middot; {SITE_HEADING}</footer>
{SCRIPT}
</body>
</html>
"""
    OUTPUT_FILE.write_text(doc, encoding="utf-8")
    print(f"Built: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
