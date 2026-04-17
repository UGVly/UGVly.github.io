#!/usr/bin/env node

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const projectRoot = path.resolve(__dirname, "..");
const notesDir = path.join(projectRoot, "notes");

function getArgValue(flag) {
  const index = process.argv.indexOf(flag);
  if (index === -1 || index === process.argv.length - 1) {
    return "";
  }

  return process.argv[index + 1];
}

function hasFlag(flag) {
  return process.argv.includes(flag);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: options.capture ? "pipe" : "inherit",
    cwd: options.cwd || projectRoot,
    env: options.env || process.env,
    encoding: "utf8"
  });

  if (result.status !== 0) {
    const message = options.capture
      ? (result.stderr || result.stdout || "").trim()
      : `${command} ${args.join(" ")} failed`;
    throw new Error(message || `${command} exited with code ${result.status}`);
  }

  return options.capture ? result.stdout.trim() : "";
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
      continue;
    }

    if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      files.push(fullPath);
    }
  }

  return files.sort();
}

function cleanMetaValue(value = "") {
  return value.trim().replace(/^['"`](.*)['"`]$/, "$1").trim();
}

function parseListValue(value = "") {
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

function parseFrontMatter(markdown) {
  const normalized = markdown.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n*/);

  if (!match) {
    return { attributes: {}, body: normalized };
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
      attributes.tags = parseListValue(rawValue).map((tag) => tag.toLowerCase());
      continue;
    }

    attributes[key] = cleanMetaValue(rawValue);
  }

  return {
    attributes,
    body: normalized.slice(match ? match[0].length : 0)
  };
}

function normalizeTags(rawTags = [], rawVisibility = "") {
  const tags = Array.isArray(rawTags)
    ? rawTags.map((tag) => String(tag).trim().toLowerCase()).filter(Boolean)
    : [];
  const unique = [];
  const seen = new Set();

  for (const tag of tags) {
    if (seen.has(tag)) {
      continue;
    }

    unique.push(tag);
    seen.add(tag);
  }

  const visibility =
    unique.includes("public") || String(rawVisibility).trim().toLowerCase() === "public"
      ? "public"
      : "private";
  const tagsWithoutVisibility = unique.filter((tag) => tag !== "public" && tag !== "private");
  return [...tagsWithoutVisibility, visibility];
}

function getVisibility(attributes) {
  const tags = normalizeTags(attributes.tags, attributes.visibility);
  return tags.includes("public") ? "public" : "private";
}

function normalizeBasePath(value = "") {
  const trimmed = String(value).trim();

  if (!trimmed || trimmed === "/") {
    return "";
  }

  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, "");
}

function deriveBasePath(repoDir) {
  const explicit = getArgValue("--base-path") || process.env.PUBLIC_SITE_BASE_PATH || "";

  if (explicit) {
    return normalizeBasePath(explicit);
  }

  const repoName = path.basename(repoDir);
  return repoName.endsWith(".github.io") ? "" : normalizeBasePath(`/${repoName}`);
}

function copyFilePreservingRelative(sourceFile, sourceRoot, targetRoot) {
  const relative = path.relative(sourceRoot, sourceFile);
  const outputPath = path.join(targetRoot, relative);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.copyFileSync(sourceFile, outputPath);
}

function resetDirectoryContents(targetDir) {
  fs.mkdirSync(targetDir, { recursive: true });

  for (const entry of fs.readdirSync(targetDir)) {
    if (entry === ".git") {
      continue;
    }

    fs.rmSync(path.join(targetDir, entry), { recursive: true, force: true });
  }
}

function ensureSafePublicNote(filePath, body) {
  const relativePath = path.relative(projectRoot, filePath);
  const riskPatterns = [
    {
      pattern: /assets\/images\/mai-private\//i,
      reason: "references private image assets"
    },
    {
      pattern: /\/Users\/|[A-Za-z]:\\\\/i,
      reason: "contains an absolute local filesystem path"
    }
  ];

  for (const { pattern, reason } of riskPatterns) {
    if (pattern.test(body)) {
      throw new Error(`Refusing to publish ${relativePath}: it ${reason}.`);
    }
  }
}

const repoDirArg = getArgValue("--repo") || process.env.PUBLIC_REPO_DIR;
const repoUrl = getArgValue("--repo-url") || process.env.PUBLIC_REPO_URL;
const remoteName = getArgValue("--remote") || process.env.PUBLIC_REPO_REMOTE || "origin";
const branchName = getArgValue("--branch") || process.env.PUBLIC_REPO_BRANCH || "main";
const commitMessage =
  getArgValue("--message") ||
  process.env.PUBLIC_REPO_COMMIT_MESSAGE ||
  `Publish public notes ${new Date().toISOString().replace("T", " ").slice(0, 19)}`;
const allowDirty = hasFlag("--allow-dirty");
const allowEmpty = hasFlag("--allow-empty");
const dryRun = hasFlag("--dry-run");

let repoDir = repoDirArg ? path.resolve(repoDirArg) : "";

if (!repoDir && !repoUrl) {
  console.error("Set --repo /path/to/public-repo or PUBLIC_REPO_DIR, or provide --repo-url.");
  process.exit(1);
}

const markdownFiles = walkMarkdown(notesDir);
const publicNotes = [];

for (const filePath of markdownFiles) {
  const markdown = fs.readFileSync(filePath, "utf8");
  const { attributes, body } = parseFrontMatter(markdown);

  if (getVisibility(attributes) !== "public") {
    continue;
  }

  ensureSafePublicNote(filePath, body);
  publicNotes.push(filePath);
}

if (publicNotes.length === 0 && !allowEmpty) {
  console.error("No public notes found. Mark at least one note with tags: [public] before publishing.");
  process.exit(1);
}

let repoDirFromClone = false;

if (!repoDir) {
  repoDir = fs.mkdtempSync(path.join(os.tmpdir(), "public-notes-repo-"));
  repoDirFromClone = true;
  run("git", ["clone", repoUrl, repoDir]);
}

if (!fs.existsSync(repoDir)) {
  if (!repoUrl) {
    console.error(`Public repo dir does not exist: ${repoDir}`);
    process.exit(1);
  }

  run("git", ["clone", repoUrl, repoDir]);
}

run("git", ["rev-parse", "--is-inside-work-tree"], { cwd: repoDir, capture: true });

if (!allowDirty) {
  const status = run("git", ["status", "--porcelain"], { cwd: repoDir, capture: true });

  if (status) {
    console.error(`Target repo has uncommitted changes: ${repoDir}`);
    process.exit(1);
  }
}

const basePath = deriveBasePath(repoDir);
const exportDir = fs.mkdtempSync(path.join(os.tmpdir(), "public-notes-export-"));
const exportNotesDir = path.join(exportDir, "notes");

for (const filePath of publicNotes) {
  copyFilePreservingRelative(filePath, notesDir, exportNotesDir);
}

run("node", [path.join(projectRoot, "build-blog.js")], {
  cwd: projectRoot,
  env: {
    ...process.env,
    BLOG_SOURCE_DIR: notesDir,
    BLOG_OUTPUT_DIR: path.join(exportDir, "blog"),
    BLOG_VISIBILITY: "public",
    BLOG_DISABLE_PRIVATE_DECOR: "1",
    SITE_BASE_PATH: basePath
  }
});

fs.writeFileSync(
  path.join(exportDir, "index.html"),
  `<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="refresh" content="0; url=./blog/index.html" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Public Notes</title>
  </head>
  <body>
    <p>Redirecting to <a href="./blog/index.html">public notes</a>…</p>
  </body>
</html>
`,
  "utf8"
);
fs.writeFileSync(path.join(exportDir, ".nojekyll"), "", "utf8");
fs.writeFileSync(
  path.join(exportDir, "README.md"),
  `# Public Notes\n\nThis repository is generated from the private notes workspace.\n\n- Published notes: ${publicNotes.length}\n- Published at: ${new Date().toISOString()}\n- Base path: ${basePath || "/"}\n`,
  "utf8"
);

if (dryRun) {
  console.log(`Dry run complete. Public export prepared at ${exportDir}`);
  process.exit(0);
}

resetDirectoryContents(repoDir);
fs.cpSync(exportDir, repoDir, { recursive: true });

run("git", ["add", "--all"], { cwd: repoDir });

const stagedStatus = run("git", ["diff", "--cached", "--name-only"], { cwd: repoDir, capture: true });

if (!stagedStatus) {
  console.log("No public-export changes to commit.");
  process.exit(0);
}

run("git", ["commit", "-m", commitMessage], { cwd: repoDir });
run("git", ["push", remoteName, branchName], { cwd: repoDir });

console.log(`Published ${publicNotes.length} public note(s) to ${repoDirFromClone ? repoUrl : repoDir}`);
