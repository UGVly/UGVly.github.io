#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const notesDir = path.resolve(process.env.NOTES_DIR || path.join(projectRoot, "notes"));
const shouldWrite = process.argv.includes("--write");

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
    return {
      hasFrontMatter: false,
      frontMatter: "",
      body: normalized
    };
  }

  return {
    hasFrontMatter: true,
    frontMatter: match[1],
    body: normalized.slice(match[0].length)
  };
}

function ensureVisibilityTags(rawTags = [], rawVisibility = "") {
  const normalizedTags = Array.isArray(rawTags)
    ? rawTags.map((tag) => String(tag).trim().toLowerCase()).filter(Boolean)
    : [];
  const unique = [];
  const seen = new Set();

  for (const tag of normalizedTags) {
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

function serializeTags(tags) {
  return `tags: [${tags.join(", ")}]`;
}

function updateMarkdown(markdown) {
  const { hasFrontMatter, frontMatter, body } = parseFrontMatter(markdown);

  if (!hasFrontMatter) {
    const nextMarkdown = `---\n${serializeTags(["private"])}\n---\n\n${body.replace(/^\n+/, "")}`;
    return {
      changed: true,
      nextMarkdown,
      visibility: "private"
    };
  }

  const lines = frontMatter.split("\n");
  const nextLines = [];
  let existingTags = [];
  let rawVisibility = "";
  let tagsLineSeen = false;

  for (const line of lines) {
    const trimmed = line.trim();
    const separatorIndex = trimmed.indexOf(":");

    if (separatorIndex === -1) {
      nextLines.push(line);
      continue;
    }

    const key = trimmed.slice(0, separatorIndex).trim().toLowerCase();
    const rawValue = trimmed.slice(separatorIndex + 1).trim();

    if (key === "tags") {
      existingTags = parseListValue(rawValue);
      tagsLineSeen = true;
      continue;
    }

    if (key === "visibility") {
      rawVisibility = cleanMetaValue(rawValue);
    }

    nextLines.push(line);
  }

  const nextTags = ensureVisibilityTags(existingTags, rawVisibility);
  const serializedTags = serializeTags(nextTags);
  const insertionIndex = nextLines.findIndex((line) => line.trim() && !line.trim().startsWith("#"));
  const targetIndex = insertionIndex === -1 ? nextLines.length : insertionIndex;
  nextLines.splice(targetIndex, 0, serializedTags);
  const nextFrontMatter = nextLines.join("\n").replace(/\n{3,}/g, "\n\n");
  const nextMarkdown = `---\n${nextFrontMatter}\n---\n\n${body.replace(/^\n+/, "")}`;

  return {
    changed: !tagsLineSeen || serializedTags !== serializeTags(existingTags),
    nextMarkdown,
    visibility: nextTags.includes("public") ? "public" : "private"
  };
}

const markdownFiles = walkMarkdown(notesDir);
let changedCount = 0;

for (const filePath of markdownFiles) {
  const original = fs.readFileSync(filePath, "utf8");
  const { changed, nextMarkdown, visibility } = updateMarkdown(original);

  if (changed && shouldWrite) {
    fs.writeFileSync(filePath, nextMarkdown, "utf8");
  }

  const prefix = changed ? (shouldWrite ? "updated" : "would-update") : "ok";
  console.log(`${prefix}\t${visibility}\t${path.relative(projectRoot, filePath)}`);

  if (changed) {
    changedCount += 1;
  }
}

console.log(
  `${shouldWrite ? "Applied" : "Detected"} visibility-tag updates for ${changedCount} note(s) in ${notesDir}`
);
