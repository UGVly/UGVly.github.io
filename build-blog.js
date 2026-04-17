const fs = require("fs");
const path = require("path");
const katex = require("katex");
const { Marked } = require("marked");

const projectRoot = __dirname;
const sourceDir = path.resolve(process.env.BLOG_SOURCE_DIR || path.join(projectRoot, "notes"));
const outputDir = path.resolve(process.env.BLOG_OUTPUT_DIR || path.join(projectRoot, "blog"));
const postsDir = path.join(outputDir, "posts");
const katexDistDir = path.join(path.dirname(require.resolve("katex/package.json")), "dist");
const katexOutputDir = path.join(outputDir, "assets", "katex");
const defaultCategory = "其他笔记";
const visibilityFilter = normalizeVisibilityFilter(process.env.BLOG_VISIBILITY || "all");
const siteBasePath = normalizeBasePath(process.env.SITE_BASE_PATH || "");
const disablePrivateDecor =
  process.env.BLOG_DISABLE_PRIVATE_DECOR === "1" || visibilityFilter === "public";

const categoryRules = [
  {
    name: "开发环境",
    keywords: [
      "zsh",
      "tmux",
      "tabby",
      "ssh",
      "shell",
      "terminal",
      "macos",
      "mac",
      "开发环境",
      "配置"
    ]
  },
  {
    name: "工程工具",
    keywords: ["github", "git", "workflow", "action", "actions", "cli", "仓库", "版本控制"]
  },
  {
    name: "研究笔记",
    keywords: [
      "reward model",
      "diffusion",
      "gaussian",
      "wigner",
      "marchenko",
      "matrix",
      "高斯",
      "矩阵",
      "推导",
      "随机矩阵",
      "偏好",
      "扩散"
    ]
  },
  {
    name: "博客说明",
    keywords: ["welcome", "notes blog", "blog", "博客", "说明"]
  }
];

function normalizeVisibilityFilter(value = "all") {
  const normalized = String(value).trim().toLowerCase();
  return ["all", "public", "private"].includes(normalized) ? normalized : "all";
}

function normalizeBasePath(value = "") {
  const trimmed = String(value).trim();

  if (!trimmed || trimmed === "/") {
    return "";
  }

  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, "");
}

function withBasePath(pathname = "/") {
  const normalizedPath = pathname.startsWith("/") ? pathname : `/${pathname}`;
  const combined = `${siteBasePath}${normalizedPath}`;
  return combined || "/";
}

function escapeHtml(value = "") {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function normalizeMarkdown(markdown) {
  return markdown.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
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

function cleanMetaValue(value = "") {
  return value.trim().replace(/^['"`](.*)['"`]$/, "$1").trim();
}

function parseListValue(value) {
  const cleaned = cleanMetaValue(value);

  if (!cleaned) {
    return [];
  }

  if (cleaned.startsWith("[") && cleaned.endsWith("]")) {
    return cleaned
      .slice(1, -1)
      .split(",")
      .map((item) => cleanMetaValue(item))
      .filter(Boolean);
  }

  return cleaned
    .split(",")
    .map((item) => cleanMetaValue(item))
    .filter(Boolean);
}

function normalizeNoteTags(rawTags = [], rawVisibility = "") {
  const tags = Array.isArray(rawTags)
    ? rawTags.map((tag) => String(tag).trim()).filter(Boolean)
    : [];
  const normalizedTags = [];
  const seen = new Set();

  for (const tag of tags) {
    const normalizedTag = tag.toLowerCase();

    if (seen.has(normalizedTag)) {
      continue;
    }

    normalizedTags.push(normalizedTag);
    seen.add(normalizedTag);
  }

  const normalizedVisibility = String(rawVisibility).trim().toLowerCase();
  const visibility =
    normalizedTags.includes("public") || normalizedVisibility === "public" ? "public" : "private";
  const tagsWithoutVisibility = normalizedTags.filter((tag) => tag !== "public" && tag !== "private");

  return [...tagsWithoutVisibility, visibility];
}

function getNoteVisibility(tags = []) {
  return tags.includes("public") ? "public" : "private";
}

function parseFrontMatter(markdown) {
  const normalized = normalizeMarkdown(markdown);
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n*/);

  if (!match) {
    return {
      attributes: {},
      body: normalized
    };
  }

  const attributes = {};

  for (const line of match[1].split("\n")) {
    const trimmed = line.trim();

    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex === -1) {
      continue;
    }

    const key = trimmed.slice(0, separatorIndex).trim().toLowerCase();
    const rawValue = trimmed.slice(separatorIndex + 1).trim();

    if (!rawValue) {
      continue;
    }

    if (key === "tags") {
      attributes.tags = parseListValue(rawValue);
      continue;
    }

    attributes[key] = cleanMetaValue(rawValue);
  }

  return {
    attributes,
    body: normalized.slice(match[0].length)
  };
}

function readTitle(markdownBody, fallback) {
  const match = markdownBody.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}

function stripLeadingTitle(markdownBody) {
  return markdownBody.replace(/^#\s+.+\n+/, "");
}

function extractDescription(markdownBody, fallback) {
  const normalized = stripLeadingTitle(markdownBody)
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\\\[[\s\S]*?\\\]/g, " ")
    .replace(/\$(.+?)\$/g, " ")
    .replace(/\\\((.+?)\\\)/g, " ")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/\[[^\]]+\]\([^)]+\)/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[`*_~]/g, " ");

  const paragraph = normalized
    .split(/\n\s*\n/)
    .map((chunk) => chunk.replace(/^#+\s+/gm, "").replace(/\s+/g, " ").trim())
    .find(Boolean);

  if (!paragraph) {
    return fallback;
  }

  return paragraph.length > 160 ? `${paragraph.slice(0, 157)}...` : paragraph;
}

function parseDateValue(value) {
  if (!value) {
    return undefined;
  }

  const normalized = String(value).trim().replace(/\//g, "-").replace("T", " ");
  const match = normalized.match(
    /^(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2})(?::(\d{1,2})(?::(\d{1,2}))?)?)?$/
  );

  if (match) {
    return new Date(
      Number(match[1]),
      Number(match[2]) - 1,
      Number(match[3]),
      Number(match[4] || 0),
      Number(match[5] || 0),
      Number(match[6] || 0)
    );
  }

  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed;
}

function extractDateHint(markdownBody) {
  const match = markdownBody.match(
    /^(?:更新时间|更新于|日期|Date)\s*[:：]\s*`?(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?)`?/m
  );

  return match ? match[1] : "";
}

function formatShortDate(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatLongDate(date) {
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`;
}

function getArchiveKey(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  return `${year}-${month}`;
}

function getArchiveLabel(date) {
  return `${date.getFullYear()} 年 ${date.getMonth() + 1} 月`;
}

function prettifyCategoryName(input) {
  const cleaned = input.replace(/[-_]+/g, " ").trim();

  if (!cleaned) {
    return defaultCategory;
  }

  if (/^[a-z0-9 ]+$/i.test(cleaned)) {
    return cleaned.replace(/\b[a-z]/g, (char) => char.toUpperCase());
  }

  return cleaned;
}

function inferCategory({ attributes, title, relativePath, markdownBody }) {
  if (attributes.category) {
    return prettifyCategoryName(attributes.category);
  }

  const normalizedPath = relativePath.replace(/\\/g, "/");
  const segments = normalizedPath.split("/");

  if (segments.length > 1) {
    return prettifyCategoryName(segments[0]);
  }

  const titleHaystack = `${title}\n${relativePath}`.toLowerCase();

  for (const rule of categoryRules) {
    if (rule.keywords.some((keyword) => titleHaystack.includes(keyword.toLowerCase()))) {
      return rule.name;
    }
  }

  const bodyHaystack = stripLeadingTitle(markdownBody).slice(0, 400).toLowerCase();

  for (const rule of categoryRules) {
    if (rule.keywords.some((keyword) => bodyHaystack.includes(keyword.toLowerCase()))) {
      return rule.name;
    }
  }

  return defaultCategory;
}

function renderMath(expression, displayMode) {
  return katex.renderToString(expression.trim(), {
    displayMode,
    throwOnError: false,
    strict: "ignore",
    output: "htmlAndMathml"
  });
}

function getCodeLanguage(lang = "") {
  return lang.trim().split(/\s+/)[0];
}

function getCodeLanguageClass(lang = "") {
  const language = getCodeLanguage(lang).toLowerCase();
  return language ? language.replace(/[^a-z0-9_-]+/g, "-") : "plain";
}

function renderCodeBlock(token) {
  const language = getCodeLanguage(token.lang);
  const languageLabel = language || "code";
  const languageClass = getCodeLanguageClass(languageLabel);
  const codeText = token.text || "";

  return `<div class="code-block" data-language="${escapeHtml(languageLabel.toLowerCase())}">
    <div class="code-block-toolbar">
      <span class="code-block-language">${escapeHtml(languageLabel)}</span>
      <button type="button" class="code-copy-button" data-state="idle" aria-label="复制代码">
        <span class="code-copy-button-label">复制</span>
      </button>
    </div>
    <pre><code class="language-${languageClass}">${escapeHtml(codeText)}</code></pre>
  </div>`;
}

function createInlineMathExtension(name, opener, closer) {
  return {
    name,
    level: "inline",
    start(src) {
      return src.indexOf(opener);
    },
    tokenizer(src) {
      if (!src.startsWith(opener)) {
        return undefined;
      }

      if (opener === "$" && src.startsWith("$$")) {
        return undefined;
      }

      const closeIndex = findInlineClose(src, opener, closer);
      if (closeIndex === -1) {
        return undefined;
      }

      const raw = src.slice(0, closeIndex + closer.length);
      const text = src.slice(opener.length, closeIndex);

      return {
        type: name,
        raw,
        text
      };
    },
    renderer(token) {
      return `<span class="math-inline">${renderMath(token.text, false)}</span>`;
    }
  };
}

function createBlockMathExtension(name, opener, closer) {
  return {
    name,
    level: "block",
    start(src) {
      return src.indexOf(opener);
    },
    tokenizer(src) {
      if (!src.startsWith(opener)) {
        return undefined;
      }

      const closeIndex = src.indexOf(closer, opener.length);
      if (closeIndex === -1) {
        return undefined;
      }

      const trailing = src.slice(closeIndex + closer.length).match(/^[ \t]*(?:\n+|$)/);
      if (!trailing) {
        return undefined;
      }

      const raw = src.slice(0, closeIndex + closer.length + trailing[0].length);
      const text = src.slice(opener.length, closeIndex).replace(/^\n/, "").replace(/\n$/, "");

      if (!text.trim()) {
        return undefined;
      }

      return {
        type: name,
        raw,
        text
      };
    },
    renderer(token) {
      if (!token.text.trim()) {
        return "";
      }

      return `<div class="math-display">${renderMath(token.text, true)}</div>`;
    }
  };
}

function findInlineClose(src, opener, closer) {
  let index = opener.length;

  while (index < src.length) {
    if (src[index] === "\n") {
      return -1;
    }

    if (src.startsWith(closer, index)) {
      if (closer === "$" && src[index - 1] === "\\") {
        index += 1;
        continue;
      }

      return index;
    }

    index += 1;
  }

  return -1;
}

const markdown = new Marked();

markdown.use({
  gfm: true,
  renderer: {
    code(token) {
      return renderCodeBlock(token);
    }
  },
  extensions: [
    createBlockMathExtension("blockMathDollar", "$$", "$$"),
    createBlockMathExtension("blockMathBracket", "\\[", "\\]"),
    createInlineMathExtension("inlineMathDollar", "$", "$"),
    createInlineMathExtension("inlineMathBracket", "\\(", "\\)")
  ]
});

function extractDisplayMath(markdownBody) {
  const lines = markdownBody.split("\n");
  const output = [];
  const blocks = [];
  let inFence = false;
  let fenceMarker = "";
  let inMath = false;
  let mathCloser = "";
  let mathBuffer = [];

  function flushMathBlock() {
    const content = mathBuffer.join("\n").trim();

    if (!content) {
      output.push("");
    } else {
      const index = blocks.push(content) - 1;
      output.push("");
      output.push(`<div data-display-math-placeholder="${index}"></div>`);
      output.push("");
    }

    inMath = false;
    mathCloser = "";
    mathBuffer = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const fenceMatch = trimmed.match(/^(```+|~~~+)/);

    if (!inMath && fenceMatch) {
      if (!inFence) {
        inFence = true;
        fenceMarker = fenceMatch[1];
      } else if (trimmed.startsWith(fenceMarker)) {
        inFence = false;
        fenceMarker = "";
      }

      output.push(line);
      continue;
    }

    if (inFence) {
      output.push(line);
      continue;
    }

    if (!inMath) {
      if (trimmed === "$$") {
        inMath = true;
        mathCloser = "$$";
        mathBuffer = [];
        continue;
      }

      if (trimmed === "\\[") {
        inMath = true;
        mathCloser = "\\]";
        mathBuffer = [];
        continue;
      }

      output.push(line);
      continue;
    }

    if (trimmed === mathCloser) {
      flushMathBlock();
      continue;
    }

    mathBuffer.push(line);
  }

  if (inMath) {
    output.push(mathCloser === "\\]" ? "\\[" : "$$");
    output.push(...mathBuffer);
  }

  return {
    body: output.join("\n"),
    blocks
  };
}

function normalizeUrlPath(value = "") {
  return String(value).replace(/\\/g, "/");
}

function splitUrlSuffix(src = "") {
  const match = String(src).match(/^([^?#]*)(.*)$/);
  return {
    pathname: match ? match[1] : String(src),
    suffix: match ? match[2] : ""
  };
}

function isExternalAssetPath(src = "") {
  return /^(?:[a-z][a-z0-9+.-]*:|\/\/|#|\/)/i.test(String(src).trim());
}

function resolveLocalAssetPath(sourceFilePath, assetPath) {
  const directPath = path.resolve(path.dirname(sourceFilePath), assetPath);

  if (fs.existsSync(directPath) && fs.statSync(directPath).isFile()) {
    return directPath;
  }

  const projectRelativePath = assetPath.replace(/^(\.\.\/|\.\/)+/, "");
  const projectPath = path.resolve(projectRoot, projectRelativePath);

  if (fs.existsSync(projectPath) && fs.statSync(projectPath).isFile()) {
    return projectPath;
  }

  return "";
}

function copyNoteAssetToOutput(assetPath) {
  const relativeFromProject = normalizeUrlPath(path.relative(projectRoot, assetPath));
  const outputRelative = relativeFromProject.startsWith("assets/")
    ? relativeFromProject.slice("assets/".length)
    : relativeFromProject.startsWith("../")
      ? path.basename(assetPath)
      : relativeFromProject;
  const targetRelative = path.posix.join("assets", "note-media", outputRelative);
  const targetPath = path.join(outputDir, targetRelative);

  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.copyFileSync(assetPath, targetPath);

  return encodeURI(withBasePath(`/blog/${targetRelative}`));
}

function rewriteLocalImageSources(html, sourceFilePath) {
  return html.replace(/<img\b([^>]*?)src=(["'])([^"']+)\2([^>]*?)>/gi, (match, before, quote, src, after) => {
    if (isExternalAssetPath(src)) {
      return match;
    }

    const { pathname, suffix } = splitUrlSuffix(src);
    const absolutePath = resolveLocalAssetPath(sourceFilePath, pathname);

    if (!absolutePath) {
      return match;
    }

    const publicUrl = `${copyNoteAssetToOutput(absolutePath)}${suffix}`;
    return `<img${before}src=${quote}${publicUrl}${quote}${after}>`;
  });
}

function renderMarkdown(markdownBody, sourceFilePath = "") {
  const stripped = stripLeadingTitle(markdownBody);
  const { body, blocks } = extractDisplayMath(stripped);
  let html = markdown.parse(body);

  html = html.replace(
    /<div data-display-math-placeholder="(\d+)"><\/div>/g,
    (_, index) => `<div class="math-display">${renderMath(blocks[Number(index)] || "", true)}</div>`
  );

  return sourceFilePath ? rewriteLocalImageSources(html, sourceFilePath) : html;
}

function ensureKatexAssets() {
  fs.mkdirSync(katexOutputDir, { recursive: true });
  fs.copyFileSync(path.join(katexDistDir, "katex.min.css"), path.join(katexOutputDir, "katex.min.css"));
  fs.cpSync(path.join(katexDistDir, "fonts"), path.join(katexOutputDir, "fonts"), {
    recursive: true,
    force: true
  });
}

function resetDirectoryContents(targetDir) {
  fs.mkdirSync(targetDir, { recursive: true });

  for (const entry of fs.readdirSync(targetDir)) {
    fs.rmSync(path.join(targetDir, entry), { recursive: true, force: true });
  }
}

function groupPostsByCategory(posts) {
  const groups = new Map();

  for (const post of posts) {
    if (!groups.has(post.category)) {
      groups.set(post.category, {
        name: post.category,
        slug: post.categorySlug,
        posts: []
      });
    }

    groups.get(post.category).posts.push(post);
  }

  return Array.from(groups.values())
    .map((group) => {
      group.posts.sort((a, b) => b.updatedAt - a.updatedAt);
      group.latestAt = group.posts[0].updatedAt;
      return group;
    })
    .sort((a, b) => {
      if (b.posts.length !== a.posts.length) {
        return b.posts.length - a.posts.length;
      }

      if (b.latestAt.getTime() !== a.latestAt.getTime()) {
        return b.latestAt - a.latestAt;
      }

      return a.name.localeCompare(b.name, "zh-CN");
    });
}

function groupPostsByArchive(posts) {
  const groups = new Map();

  for (const post of posts) {
    if (!groups.has(post.archiveKey)) {
      groups.set(post.archiveKey, {
        key: post.archiveKey,
        slug: slugify(post.archiveKey) || "archive",
        label: post.archiveLabel,
        sortDate: new Date(post.updatedAt.getFullYear(), post.updatedAt.getMonth(), 1),
        posts: []
      });
    }

    groups.get(post.archiveKey).posts.push(post);
  }

  return Array.from(groups.values())
    .map((group) => {
      group.posts.sort((a, b) => b.updatedAt - a.updatedAt);
      return group;
    })
    .sort((a, b) => b.sortDate - a.sortDate);
}

function renderMetaBadge(label, href = "") {
  const content = `<span>${escapeHtml(label)}</span>`;
  return href ? `<a class="meta-badge" href="${href}">${content}</a>` : `<span class="meta-badge">${content}</span>`;
}

function renderPostCard(post) {
  return `<article class="post-card">
    <div class="post-card-meta">
      ${renderMetaBadge(post.category, `${withBasePath("/blog/index.html")}#category-${post.categorySlug}`)}
      <span class="meta-inline">${escapeHtml(post.updatedLabel)}</span>
    </div>
    <h3><a href="${withBasePath(`/blog/posts/${post.slug}.html`)}">${escapeHtml(post.title)}</a></h3>
    <p>${escapeHtml(post.description)}</p>
    <div class="post-card-footer">
      <span>${escapeHtml(post.archiveLabel)}</span>
      <span>${escapeHtml(post.relativePath)}</span>
    </div>
  </article>`;
}

function renderCategoryNavigation(groups) {
  return `<div class="jump-list">${groups
    .map(
      (group) => `<a class="jump-chip" href="#category-${group.slug}">
          <span>${escapeHtml(group.name)}</span>
          <strong>${group.posts.length}</strong>
        </a>`
    )
    .join("\n")}</div>`;
}

function renderArchiveNavigation(groups) {
  return `<div class="jump-list">${groups
    .map(
      (group) => `<a class="jump-chip jump-chip-muted" href="#archive-${group.slug}">
          <span>${escapeHtml(group.label)}</span>
          <strong>${group.posts.length}</strong>
        </a>`
    )
    .join("\n")}</div>`;
}

function renderCategorySections(groups) {
  return groups
    .map(
      (group) => `<section class="collection-block" id="category-${group.slug}">
          <div class="collection-header">
            <div>
              <p class="section-label">Category</p>
              <h2>${escapeHtml(group.name)}</h2>
            </div>
            <span class="collection-count">${group.posts.length} 篇</span>
          </div>
          <div class="post-grid">
            ${group.posts.map((post) => renderPostCard(post)).join("\n")}
          </div>
        </section>`
    )
    .join("\n");
}

function renderArchiveSections(groups) {
  return groups
    .map(
      (group) => `<section class="collection-block archive-block" id="archive-${group.slug}">
          <div class="collection-header">
            <div>
              <p class="section-label">Archive</p>
              <h2>${escapeHtml(group.label)}</h2>
            </div>
            <span class="collection-count">${group.posts.length} 篇</span>
          </div>
          <div class="archive-list">
            ${group.posts
              .map(
                (post) => `<article class="archive-item">
                    <div>
                      <div class="archive-item-meta">
                        ${renderMetaBadge(post.category, `${withBasePath("/blog/index.html")}#category-${post.categorySlug}`)}
                      </div>
                      <h3><a href="${withBasePath(`/blog/posts/${post.slug}.html`)}">${escapeHtml(post.title)}</a></h3>
                      <p>${escapeHtml(post.description)}</p>
                    </div>
                    <div class="archive-item-side">
                      <time datetime="${escapeHtml(post.updatedISO)}">${escapeHtml(post.updatedLabel)}</time>
                      <span>${escapeHtml(post.relativePath)}</span>
                    </div>
                  </article>`
              )
              .join("\n")}
          </div>
        </section>`
    )
    .join("\n");
}

function renderDashboard(posts, categoryGroups, archiveGroups) {
  const latestPost = posts[0];

  return `<section class="dashboard-grid">
    <article class="dashboard-card">
      <span class="dashboard-label">文章总数</span>
      <strong>${posts.length}</strong>
    </article>
    <article class="dashboard-card">
      <span class="dashboard-label">分类数量</span>
      <strong>${categoryGroups.length}</strong>
    </article>
    <article class="dashboard-card">
      <span class="dashboard-label">归档月份</span>
      <strong>${archiveGroups.length}</strong>
    </article>
    <article class="dashboard-card">
      <span class="dashboard-label">最近更新</span>
      <strong>${latestPost ? escapeHtml(latestPost.updatedLabel) : "暂无"}</strong>
    </article>
  </section>`;
}

function renderPostHeaderMeta(post) {
  const displayTags = post.tags.filter((tag) => tag !== "public" && tag !== "private");
  const tagBadges = displayTags.map((tag) => renderMetaBadge(`#${tag}`)).join("");

  return `<div class="post-header-meta">
    ${renderMetaBadge(post.category, `${withBasePath("/blog/index.html")}#category-${post.categorySlug}`)}
    ${tagBadges}
    <span class="meta-inline">更新于 ${escapeHtml(post.updatedLabel)}</span>
    <span class="meta-inline">归档 ${escapeHtml(post.archiveLabel)}</span>
  </div>
  <p class="source-note">来源文件：<code>${escapeHtml(post.relativePath)}</code></p>`;
}

function getHeroPanelBeforeBackground() {
  if (disablePrivateDecor) {
    return `
          linear-gradient(90deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.08)),
          radial-gradient(circle at 30% 50%, rgba(142, 166, 255, 0.16), rgba(142, 166, 255, 0)),
          radial-gradient(circle at 72% 40%, rgba(210, 108, 153, 0.14), rgba(210, 108, 153, 0))
    `.trim();
  }

  return `
          linear-gradient(90deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.08)),
          linear-gradient(180deg, rgba(16, 19, 33, 0.06), rgba(16, 19, 33, 0.14)),
          url("${withBasePath("/assets/images/mai-private/mai-fanofanime2.png")}")
  `.trim();
}

function getContentPanelBeforeBackground() {
  if (disablePrivateDecor) {
    return `
          radial-gradient(circle, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0)),
          radial-gradient(circle at 62% 42%, rgba(178, 160, 255, 0.16), rgba(178, 160, 255, 0))
    `.trim();
  }

  return `
          radial-gradient(circle, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0)),
          url("${withBasePath("/assets/images/mai-private/mai-closeup.jpg")}")
  `.trim();
}

function pageTemplate({
  title,
  description,
  heroEyebrow = "Notes / 笔记",
  heroKicker = "",
  heroTitle = title,
  heroDescription = description,
  content,
  listHtml = ""
}) {
  const heroKickerHtml = heroKicker
    ? `<p class="hero-kicker">${escapeHtml(heroKicker)}</p>`
    : "";
  const heroDescriptionHtml = heroDescription
    ? `<p class="hero-description">${escapeHtml(heroDescription)}</p>`
    : "";

  return `<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(title)}</title>
    <meta name="description" content="${escapeHtml(description)}" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Noto+Serif:wght@500;700&family=Noto+Serif+SC:wght@500;700&display=swap"
      rel="stylesheet"
    />
    <link rel="stylesheet" href="${withBasePath("/blog/assets/katex/katex.min.css")}" />
    <style>
      :root {
        --bg: #f4f0fb;
        --panel: rgba(255, 251, 255, 0.78);
        --panel-strong: #fffbff;
        --panel-soft: rgba(255, 255, 255, 0.34);
        --text: #232238;
        --muted: #615f78;
        --line: rgba(65, 61, 103, 0.12);
        --accent: #6f63c8;
        --accent-2: #d26c99;
        --accent-3: #8ea6ff;
        --accent-soft: rgba(111, 99, 200, 0.12);
        --shadow: 0 28px 72px rgba(76, 67, 125, 0.16);
        --code-bg: #13192d;
        --code-text: #f7f4ff;
      }
      body.dark {
        --bg: #141726;
        --panel: rgba(21, 24, 40, 0.8);
        --panel-strong: #171a2c;
        --panel-soft: rgba(24, 27, 44, 0.46);
        --text: #f7f4ff;
        --muted: #c7c0df;
        --line: rgba(247, 244, 255, 0.12);
        --accent: #b5a1ff;
        --accent-2: #ff9fc7;
        --accent-3: #8fb4ff;
        --accent-soft: rgba(181, 161, 255, 0.16);
        --shadow: 0 26px 70px rgba(0, 0, 0, 0.38);
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(210, 108, 153, 0.18), transparent 28%),
          radial-gradient(circle at 82% 12%, rgba(142, 166, 255, 0.22), transparent 26%),
          radial-gradient(circle at 50% 100%, rgba(111, 99, 200, 0.12), transparent 35%),
          linear-gradient(180deg, #f9f5ff 0%, var(--bg) 48%, #e9e4f6 100%);
        font-family: "Manrope", sans-serif;
        transition: background 240ms ease, color 240ms ease;
      }
      body.dark {
        background:
          radial-gradient(circle at top left, rgba(255, 159, 199, 0.14), transparent 24%),
          radial-gradient(circle at 88% 15%, rgba(143, 180, 255, 0.18), transparent 24%),
          radial-gradient(circle at 45% 100%, rgba(181, 161, 255, 0.12), transparent 30%),
          linear-gradient(180deg, #101321 0%, var(--bg) 50%, #171a2b 100%);
      }
      a {
        color: inherit;
        text-decoration: none;
      }
      a:hover {
        text-decoration: none;
      }
      img {
        display: block;
        max-width: 100%;
      }
      .shell {
        width: min(1120px, calc(100% - 32px));
        margin: 0 auto;
        padding: 24px 0 80px;
      }
      main {
        display: grid;
        gap: 24px;
      }
      .site-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        padding: 8px 0 28px;
      }
      .brand {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px;
        height: 52px;
        border: 1px solid var(--line);
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.78), rgba(221, 213, 255, 0.58));
        backdrop-filter: blur(12px);
        font-weight: 800;
        letter-spacing: 0.08em;
      }
      .site-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        color: var(--muted);
      }
      .site-nav a {
        padding: 0.7rem 1rem;
        border: 1px solid transparent;
        border-radius: 999px;
        transition: border-color 180ms ease, background 180ms ease, color 180ms ease;
      }
      .site-nav a:hover {
        border-color: var(--line);
        background: rgba(255, 255, 255, 0.28);
        color: var(--text);
      }
      .theme-toggle {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 30;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 14px 18px;
        background: var(--panel-strong);
        color: var(--text);
        font: inherit;
        font-weight: 700;
        cursor: pointer;
        box-shadow: var(--shadow);
      }
      .blog-hero {
        display: block;
      }
      .hero-panel,
      .content-panel {
        border: 1px solid var(--line);
        border-radius: 28px;
        background: var(--panel);
        backdrop-filter: blur(20px);
        box-shadow: var(--shadow);
      }
      .hero-panel {
        position: relative;
        overflow: hidden;
        padding: 34px;
        background:
          radial-gradient(circle at top right, rgba(255, 255, 255, 0.82), transparent 30%),
          linear-gradient(180deg, rgba(255, 255, 255, 0.26), transparent 45%),
          var(--panel);
      }
      .hero-panel::before {
        content: "";
        position: absolute;
        top: 24px;
        right: -40px;
        width: 42%;
        height: calc(100% - 48px);
        border-radius: 28px 0 0 28px;
        background: ${getHeroPanelBeforeBackground()};
        background-size: cover;
        background-position: 62% center;
        opacity: 0.16;
        filter: saturate(0.92);
        pointer-events: none;
      }
      .hero-panel::after {
        content: "";
        position: absolute;
        right: -86px;
        bottom: -104px;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(178, 160, 255, 0.22), rgba(178, 160, 255, 0));
        pointer-events: none;
      }
      .eyebrow {
        margin: 0 0 12px;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.78rem;
        font-weight: 700;
      }
      .hero-kicker {
        margin: 0 0 18px;
        color: var(--accent-2);
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      h1, h2, h3, h4, .brand {
        margin: 0;
        color: var(--text);
        line-height: 1.2;
        font-family: "Noto Serif", serif;
      }
      html:lang(zh-CN) h1,
      html:lang(zh-CN) h2,
      html:lang(zh-CN) h3,
      html:lang(zh-CN) h4,
      html:lang(zh-CN) .brand {
        font-family: "Noto Serif SC", serif;
      }
      h1 {
        font-size: clamp(2.3rem, 6vw, 4.7rem);
        line-height: 1.02;
      }
      h2 {
        font-size: clamp(1.5rem, 3vw, 2.3rem);
      }
      .hero-description {
        max-width: 56ch;
        margin: 20px 0 0;
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.82;
      }
      .content-panel {
        position: relative;
        overflow: hidden;
        padding: 34px;
      }
      .content-panel::before {
        content: "";
        position: absolute;
        top: 30px;
        right: -120px;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: ${getContentPanelBeforeBackground()};
        background-size: cover;
        background-position: 48% center;
        opacity: 0.09;
        filter: blur(2px) saturate(0.85);
        pointer-events: none;
      }
      .content-panel > * {
        position: relative;
        z-index: 1;
      }
      .content-panel > :first-child {
        margin-top: 0;
      }
      .note-intro {
        max-width: 70ch;
      }
      .note-article {
        max-width: 78ch;
      }
      .note-content > :first-child {
        margin-top: 0;
      }
      .note-content h2,
      .note-content h3,
      .note-content h4 {
        margin-top: 2.4rem;
        margin-bottom: 0.9rem;
      }
      .note-content p,
      .note-content li,
      .note-content blockquote {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.9;
      }
      .note-content a {
        color: var(--accent);
      }
      .note-content ul,
      .note-content ol {
        margin: 1rem 0;
        padding-left: 1.5rem;
      }
      .note-content li + li {
        margin-top: 0.5rem;
      }
      .note-content hr {
        border: 0;
        border-top: 1px solid var(--line);
        margin: 2.2rem 0;
      }
      .note-content blockquote {
        margin: 1.5rem 0;
        padding: 0.35rem 1rem;
        border-left: 4px solid rgba(210, 108, 153, 0.35);
        border-radius: 0 18px 18px 0;
        background: rgba(255, 255, 255, 0.42);
      }
      .note-content img {
        margin: 1.7rem 0;
        border-radius: 22px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
      }
      .note-content pre {
        margin: 1.5rem 0;
        padding: 16px;
        border-radius: 18px;
        background: var(--code-bg);
        color: var(--code-text);
      }
      .note-content .code-block {
        margin: 1.5rem 0;
        border-radius: 18px;
        overflow: hidden;
        background: var(--code-bg);
        color: var(--code-text);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
      }
      .note-content .code-block-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
      }
      .note-content .code-block-language {
        min-width: 0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(246, 241, 232, 0.68);
      }
      .note-content .code-copy-button {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.5rem 0.85rem;
        background: rgba(255, 255, 255, 0.08);
        color: var(--code-text);
        font: inherit;
        font-size: 0.88rem;
        font-weight: 700;
        cursor: pointer;
        transition: background 160ms ease, border-color 160ms ease, transform 160ms ease, color 160ms ease;
      }
      .note-content .code-copy-button:hover {
        background: rgba(255, 255, 255, 0.14);
        border-color: rgba(255, 255, 255, 0.22);
      }
      .note-content .code-copy-button:active {
        transform: translateY(1px);
      }
      .note-content .code-copy-button[data-state="copied"] {
        border-color: rgba(111, 99, 200, 0.48);
        background: rgba(111, 99, 200, 0.24);
        color: #f3eeff;
      }
      .note-content .code-copy-button[data-state="failed"] {
        border-color: rgba(210, 108, 153, 0.48);
        background: rgba(210, 108, 153, 0.18);
      }
      .note-content .code-block pre {
        overflow: auto;
        margin: 0;
        padding: 18px 20px 20px;
        border-radius: 0;
        background: transparent;
      }
      .note-content code,
      .dashboard-card code {
        padding: 0.15em 0.35em;
        border-radius: 6px;
        background: rgba(35, 34, 56, 0.08);
      }
      .note-content pre code {
        padding: 0;
        background: transparent;
        color: inherit;
      }
      .note-content table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
      }
      .note-content th,
      .note-content td {
        border: 1px solid var(--line);
        padding: 0.75rem;
        text-align: left;
      }
      .note-content th {
        background: rgba(255, 255, 255, 0.3);
      }
      .math-inline .katex {
        font-size: 1.04em;
      }
      .math-display {
        margin: 1.4rem 0;
        overflow-x: auto;
        overflow-y: hidden;
      }
      .math-display .katex-display {
        margin: 0;
        padding: 0.8rem 1rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.32);
      }
      .post-header-meta,
      .post-card-meta,
      .archive-item-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }
      .post-header-meta {
        margin-bottom: 10px;
      }
      .meta-badge {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(111, 99, 200, 0.2);
        border-radius: 999px;
        padding: 0.34rem 0.7rem;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.84rem;
        font-weight: 700;
      }
      .meta-inline {
        color: var(--muted);
        font-size: 0.92rem;
      }
      .source-note {
        margin: 0 0 24px;
        color: var(--muted);
        font-size: 0.95rem;
      }
      .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 14px;
        margin: 28px 0 10px;
      }
      .dashboard-card,
      .post-card,
      .collection-block,
      .archive-item {
        border: 1px solid var(--line);
        border-radius: 24px;
        background: var(--panel-soft);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
      }
      .dashboard-card {
        padding: 18px;
      }
      .dashboard-label,
      .section-label {
        display: block;
        margin-bottom: 10px;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        font-weight: 700;
      }
      .dashboard-card strong {
        display: block;
        font-family: "Noto Serif", serif;
        font-size: 1.8rem;
      }
      html:lang(zh-CN) .dashboard-card strong {
        font-family: "Noto Serif SC", serif;
      }
      .dashboard-card p,
      .post-card p,
      .collection-intro,
      .archive-item p {
        margin: 10px 0 0;
        color: var(--muted);
        line-height: 1.7;
      }
      .section-stack {
        display: grid;
        gap: 28px;
        margin-top: 24px;
      }
      .section-shell {
        padding-top: 8px;
      }
      .section-header {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-bottom: 18px;
      }
      .jump-list {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 0 0 18px;
      }
      .jump-chip {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 0.65rem 0.9rem;
        border: 1px solid rgba(111, 99, 200, 0.18);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.42);
        color: var(--text);
        transition: border-color 180ms ease, transform 180ms ease, background 180ms ease;
      }
      .jump-chip:hover {
        border-color: rgba(111, 99, 200, 0.32);
        background: rgba(255, 255, 255, 0.62);
        transform: translateY(-1px);
      }
      .jump-chip strong {
        color: var(--accent);
        font-size: 0.92rem;
      }
      .jump-chip-muted {
        border-color: var(--line);
      }
      .collection-block {
        padding: 18px;
      }
      .collection-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 16px;
      }
      .collection-count {
        flex-shrink: 0;
        padding: 0.36rem 0.75rem;
        border-radius: 999px;
        background: rgba(111, 99, 200, 0.1);
        color: var(--accent);
        font-size: 0.88rem;
        font-weight: 700;
      }
      .post-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 14px;
      }
      .post-card {
        padding: 18px;
        transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
      }
      .post-card:hover {
        transform: translateY(-2px);
        border-color: rgba(111, 99, 200, 0.24);
        background: rgba(255, 255, 255, 0.5);
      }
      .post-card h3,
      .archive-item h3 {
        margin-top: 14px;
        font-size: 1.12rem;
      }
      .post-card h3 a,
      .archive-item h3 a {
        color: var(--text);
      }
      .post-card h3 a:hover,
      .archive-item h3 a:hover {
        color: var(--accent);
      }
      .post-card-footer {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 12px;
        margin-top: 16px;
        color: var(--muted);
        font-size: 0.86rem;
      }
      .archive-list {
        display: grid;
        gap: 12px;
      }
      .archive-item {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 16px;
        padding: 18px;
      }
      .archive-item-side {
        display: flex;
        flex-direction: column;
        gap: 6px;
        align-items: flex-end;
        color: var(--muted);
        font-size: 0.86rem;
        white-space: nowrap;
      }
      @media (max-width: 980px) {
      }
      @media (max-width: 900px) {
        .archive-item {
          grid-template-columns: 1fr;
        }
        .archive-item-side {
          align-items: flex-start;
          white-space: normal;
        }
      }
      @media (max-width: 720px) {
        .shell {
          width: min(100% - 16px, 1120px);
          padding-top: 16px;
        }
        .site-header {
          flex-direction: column;
          align-items: flex-start;
        }
        .hero-panel,
        .content-panel {
          padding: 20px;
          border-radius: 24px;
        }
        .hero-panel::before {
          width: 44%;
          right: -48px;
          height: calc(100% - 40px);
          top: 20px;
          opacity: 0.11;
        }
        .collection-block,
        .post-card,
        .dashboard-card {
          padding: 16px;
        }
        .note-content p,
        .note-content li,
        .note-content blockquote {
          font-size: 1rem;
        }
        .note-content .code-block-toolbar {
          padding: 10px 12px;
        }
        .note-content .code-copy-button {
          padding: 0.45rem 0.72rem;
        }
        .note-content .code-block pre {
          padding: 16px;
        }
        .collection-header {
          flex-direction: column;
        }
        .jump-chip {
          width: 100%;
          justify-content: space-between;
        }
        .theme-toggle {
          left: 12px;
          right: 12px;
          bottom: 12px;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="site-header">
        <a class="brand" href="${withBasePath("/")}">ZJ</a>
        <nav class="site-nav">
          <a href="${withBasePath("/")}">Home</a>
          <a href="${withBasePath("/#research")}">Research</a>
          <a href="${withBasePath("/blog/index.html")}">Notes</a>
        </nav>
      </header>
      <main>
        <section class="blog-hero">
          <div class="hero-panel">
            <p class="eyebrow">${escapeHtml(heroEyebrow)}</p>
            ${heroKickerHtml}
            <h1>${escapeHtml(heroTitle)}</h1>
            ${heroDescriptionHtml}
          </div>
        </section>
        <section class="content-panel">
        ${content}
        ${listHtml}
        </section>
      </main>
    </div>
    <button class="theme-toggle" type="button" aria-label="Toggle theme">
      切换配色
    </button>
    <script>
      (function () {
        const themeStorageKey = "personal-homepage-theme";
        const themeToggle = document.querySelector(".theme-toggle");

        function applyTheme(theme) {
          document.body.classList.toggle("dark", theme === "dark");
        }

        const savedTheme = localStorage.getItem(themeStorageKey);
        if (savedTheme) {
          applyTheme(savedTheme);
        }

        function fallbackCopyText(text) {
          return new Promise(function (resolve, reject) {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.setAttribute("readonly", "");
            textarea.style.position = "fixed";
            textarea.style.top = "-9999px";
            textarea.style.opacity = "0";
            document.body.appendChild(textarea);
            textarea.select();
            textarea.setSelectionRange(0, textarea.value.length);

            try {
              const succeeded = document.execCommand("copy");
              document.body.removeChild(textarea);

              if (succeeded) {
                resolve();
                return;
              }

              reject(new Error("execCommand copy failed"));
            } catch (error) {
              document.body.removeChild(textarea);
              reject(error);
            }
          });
        }

        function copyText(text) {
          if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
            return navigator.clipboard.writeText(text).catch(function () {
              return fallbackCopyText(text);
            });
          }

          return fallbackCopyText(text);
        }

        function setButtonState(button, state) {
          const label = button.querySelector(".code-copy-button-label");
          const textMap = {
            idle: "复制",
            copied: "已复制",
            failed: "重试"
          };

          button.dataset.state = state;
          label.textContent = textMap[state];
        }

        document.addEventListener("click", function (event) {
          const button = event.target.closest(".code-copy-button");

          if (!button) {
            return;
          }

          const block = button.closest(".code-block");
          const code = block && block.querySelector("pre code");

          if (!code) {
            return;
          }

          const text = code.textContent || "";
          window.clearTimeout(button._resetTimer);
          button.disabled = true;

          copyText(text)
            .then(function () {
              setButtonState(button, "copied");
            })
            .catch(function () {
              setButtonState(button, "failed");
            })
            .finally(function () {
              button._resetTimer = window.setTimeout(function () {
                setButtonState(button, "idle");
              button.disabled = false;
              }, 1800);
            });
        });

        if (themeToggle) {
          themeToggle.addEventListener("click", function () {
            const nextTheme = document.body.classList.contains("dark") ? "light" : "dark";
            applyTheme(nextTheme);
            localStorage.setItem(themeStorageKey, nextTheme);
          });
        }
      })();
    </script>
  </body>
</html>`;
}

resetDirectoryContents(outputDir);
fs.mkdirSync(postsDir, { recursive: true });
ensureKatexAssets();

const markdownFiles = walkMarkdown(sourceDir);
const posts = [];

for (const filePath of markdownFiles) {
  const rawSource = fs.readFileSync(filePath, "utf8");
  const { attributes, body } = parseFrontMatter(rawSource);
  const relativePath = path.relative(sourceDir, filePath).replace(/\\/g, "/");
  const baseName = relativePath.replace(/\.md$/i, "");
  const slug = slugify(baseName) || "post";
  const title = readTitle(body, path.basename(baseName));
  const description = extractDescription(body, title);
  const outputPath = path.join(postsDir, `${slug}.html`);
  const stat = fs.statSync(filePath);
  const derivedDate = parseDateValue(attributes.date) || parseDateValue(extractDateHint(body)) || stat.mtime;
  const category = inferCategory({
    attributes,
    title,
    relativePath,
    markdownBody: body
  });
  const categorySlug = slugify(category) || "category";
  const tags = normalizeNoteTags(attributes.tags, attributes.visibility);
  const visibility = getNoteVisibility(tags);

  if (visibilityFilter !== "all" && visibility !== visibilityFilter) {
    continue;
  }

  const content = renderMarkdown(body, filePath);

  const post = {
    title,
    slug,
    relativePath,
    description,
    category,
    categorySlug,
    tags,
    visibility,
    updatedAt: derivedDate,
    updatedISO: formatShortDate(derivedDate),
    updatedLabel: formatLongDate(derivedDate),
    archiveKey: getArchiveKey(derivedDate),
    archiveLabel: getArchiveLabel(derivedDate)
  };

  const article = pageTemplate({
    title,
    description,
    heroTitle: title,
    heroDescription: description,
    content: `<article class="note-article">
      ${renderPostHeaderMeta(post)}
      <div class="note-content">${content}</div>
    </article>`
  });

  fs.writeFileSync(outputPath, article, "utf8");
  posts.push(post);
}

posts.sort((a, b) => b.updatedAt - a.updatedAt);

const categoryGroups = groupPostsByCategory(posts);
const archiveGroups = groupPostsByArchive(posts);

const listHtml = posts.length
  ? `<div class="section-stack">
      ${renderDashboard(posts, categoryGroups, archiveGroups)}
      <section class="section-shell">
        <div class="section-header">
          <div>
            <p class="section-label">Categories</p>
            <h2>按分类查看</h2>
          </div>
        </div>
        ${renderCategoryNavigation(categoryGroups)}
        ${renderCategorySections(categoryGroups)}
      </section>
      <section class="section-shell">
        <div class="section-header">
          <div>
            <p class="section-label">Archive</p>
            <h2>按时间归档</h2>
          </div>
        </div>
        ${renderArchiveNavigation(archiveGroups)}
        ${renderArchiveSections(archiveGroups)}
      </section>
    </div>`
  : `<p>No markdown notes found. Put your files in <code>${escapeHtml(sourceDir)}</code> and run <code>npm run build:blog</code>.</p>`;

const indexHtml = pageTemplate({
  title: "Notes Blog",
  description: "Markdown notes blog with category sections and archive navigation.",
  heroTitle: "Notes Blog",
  heroDescription: "",
  content: "",
  listHtml
});

fs.writeFileSync(path.join(outputDir, "index.html"), indexHtml, "utf8");

console.log(
  `Built ${posts.length} blog post(s) from ${sourceDir} (visibility filter: ${visibilityFilter})`
);
