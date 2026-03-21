const fs = require("fs");
const path = require("path");

const projectRoot = __dirname;
const sourceDir = path.resolve(process.env.BLOG_SOURCE_DIR || path.join(projectRoot, "notes"));
const outputDir = path.join(projectRoot, "blog");
const postsDir = path.join(outputDir, "posts");

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function slugify(input) {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, "-")
    .replace(/^-+|-+$/g, "");
}

function walkMarkdown(dir) {
  if (!fs.existsSync(dir)) {
    return [];
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkMarkdown(fullPath));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      files.push(fullPath);
    }
  }

  return files;
}

function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inList = false;
  let inCode = false;
  let paragraph = [];

  function flushParagraph() {
    if (paragraph.length) {
      html.push(`<p>${formatInline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  }

  function closeList() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  }

  for (const line of lines) {
    if (line.startsWith("```")) {
      flushParagraph();
      closeList();
      if (!inCode) {
        html.push("<pre><code>");
        inCode = true;
      } else {
        html.push("</code></pre>");
        inCode = false;
      }
      continue;
    }

    if (inCode) {
      html.push(`${escapeHtml(line)}\n`);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${formatInline(heading[2])}</h${level}>`);
      continue;
    }

    const list = line.match(/^[-*]\s+(.*)$/);
    if (list) {
      flushParagraph();
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${formatInline(list[1])}</li>`);
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  closeList();
  if (inCode) {
    html.push("</code></pre>");
  }

  return html.join("\n");
}

function readTitle(markdown, fallback) {
  const match = markdown.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}

function pageTemplate({ title, description, content, listHtml = "" }) {
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(title)}</title>
    <meta name="description" content="${escapeHtml(description)}" />
    <style>
      :root {
        --bg: #f4efe6;
        --panel: #fff9f0;
        --text: #1f1a16;
        --muted: #655a52;
        --line: rgba(31, 26, 22, 0.12);
        --accent: #c44f2a;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--text);
        background: linear-gradient(180deg, #efe4d1 0%, var(--bg) 100%);
        font-family: "Segoe UI", sans-serif;
      }
      .shell {
        width: min(900px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 80px;
      }
      .panel {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 28px;
        background: rgba(255, 249, 240, 0.92);
      }
      a { color: var(--accent); text-decoration: none; }
      .nav {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        color: var(--muted);
      }
      .eyebrow {
        margin: 0 0 12px;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        font-weight: 700;
      }
      h1, h2, h3 { line-height: 1.2; }
      p, li { color: var(--muted); line-height: 1.8; }
      pre {
        overflow: auto;
        padding: 16px;
        border-radius: 16px;
        background: #201d1a;
        color: #f6f1e8;
      }
      code {
        padding: 0.15em 0.35em;
        border-radius: 6px;
        background: rgba(31, 26, 22, 0.08);
      }
      pre code { background: transparent; padding: 0; }
      .post-list {
        display: grid;
        gap: 14px;
        margin-top: 24px;
      }
      .post-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        background: rgba(255, 255, 255, 0.4);
      }
      .post-card h2 {
        margin: 0 0 8px;
        font-size: 1.2rem;
      }
      .post-card p {
        margin: 0;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <nav class="nav">
        <a href="/">Home</a>
        <a href="/blog/index.html">Blog</a>
      </nav>
      <section class="panel">
        ${content}
        ${listHtml}
      </section>
    </div>
  </body>
</html>`;
}

fs.mkdirSync(postsDir, { recursive: true });

const markdownFiles = walkMarkdown(sourceDir);
const posts = [];

for (const filePath of markdownFiles) {
  const markdown = fs.readFileSync(filePath, "utf8");
  const relativePath = path.relative(sourceDir, filePath);
  const baseName = relativePath.replace(/\\/g, "/").replace(/\.md$/i, "");
  const slug = slugify(baseName) || "post";
  const title = readTitle(markdown, path.basename(baseName));
  const outputPath = path.join(postsDir, `${slug}.html`);
  const content = markdownToHtml(markdown);
  const article = pageTemplate({
    title,
    description: title,
    content: `<p class="eyebrow">Notes / 笔记</p><h1>${escapeHtml(title)}</h1>${content}`
  });

  fs.writeFileSync(outputPath, article, "utf8");
  posts.push({
    title,
    slug,
    relativePath,
    updatedAt: fs.statSync(filePath).mtime
  });
}

posts.sort((a, b) => b.updatedAt - a.updatedAt);

const listHtml = posts.length
  ? `<div class="post-list">${posts
      .map(
        (post) => `<article class="post-card">
            <h2><a href="/blog/posts/${post.slug}.html">${escapeHtml(post.title)}</a></h2>
            <p>${escapeHtml(post.relativePath)}</p>
          </article>`
      )
      .join("\n")}</div>`
  : `<p>No markdown notes found. Put your files in <code>${escapeHtml(sourceDir)}</code> and run <code>npm.cmd run build:blog</code>.</p>`;

const indexHtml = pageTemplate({
  title: "Notes Blog",
  description: "Markdown notes transformed into a simple blog.",
  content:
    `<p class="eyebrow">Notes / 笔记</p><h1>Notes Blog</h1><p>Place your markdown notes in <code>${escapeHtml(
      sourceDir
    )}</code>, then run <code>npm.cmd run build:blog</code> to regenerate this section of the site.</p>`,
  listHtml
});

fs.writeFileSync(path.join(outputDir, "index.html"), indexHtml, "utf8");

console.log(`Built ${posts.length} blog post(s) from ${sourceDir}`);
