const http = require("http");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const localBlogRoot = path.join(root, ".local", "blog");
const port = process.env.PORT || 3000;

const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".ico": "image/x-icon"
};

function sendFile(filePath, res) {
  fs.stat(filePath, (statErr, stats) => {
    if (statErr) {
      res.writeHead(statErr.code === "ENOENT" ? 404 : 500, {
        "Content-Type": "text/plain; charset=utf-8"
      });
      res.end(statErr.code === "ENOENT" ? "404 Not Found" : "500 Server Error");
      return;
    }

    const resolvedPath = stats.isDirectory() ? path.join(filePath, "index.html") : filePath;

    fs.readFile(resolvedPath, (err, data) => {
      if (err) {
        res.writeHead(err.code === "ENOENT" ? 404 : 500, {
          "Content-Type": "text/plain; charset=utf-8"
        });
        res.end(err.code === "ENOENT" ? "404 Not Found" : "500 Server Error");
        return;
      }

      const ext = path.extname(resolvedPath).toLowerCase();
      res.writeHead(200, {
        "Content-Type": types[ext] || "application/octet-stream"
      });
      res.end(data);
    });
  });
}

function normalizeRequestPath(rawUrl) {
  if (rawUrl === "/") {
    return "/index.html";
  }

  if (rawUrl === "/blog") {
    return "/blog/index.html";
  }

  return rawUrl.endsWith("/") ? `${rawUrl}index.html` : rawUrl;
}

function resolveRequestTarget(urlPath) {
  const useLocalBlog =
    fs.existsSync(localBlogRoot) && /^\/blog(?:\/|$)/.test(urlPath);

  if (useLocalBlog) {
    const localPath = urlPath.replace(/^\/blog/, "") || "/index.html";
    return {
      baseRoot: localBlogRoot,
      relativePath: localPath
    };
  }

  return {
    baseRoot: root,
    relativePath: urlPath
  };
}

http
  .createServer((req, res) => {
    const rawUrl = decodeURIComponent((req.url || "/").split("?")[0]);
    const normalizedUrl = normalizeRequestPath(rawUrl);
    const { baseRoot, relativePath } = resolveRequestTarget(normalizedUrl);
    const safePath = path.normalize(relativePath).replace(/^(\.\.[/\\])+/, "");
    const filePath = path.join(baseRoot, safePath);

    if (!filePath.startsWith(baseRoot)) {
      res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("403 Forbidden");
      return;
    }

    sendFile(filePath, res);
  })
  .listen(port, () => {
    const blogSource = fs.existsSync(localBlogRoot) ? ".local/blog (all notes)" : "blog (public fallback)";
    console.log(`Personal homepage running at http://localhost:${port}`);
    console.log(`Serving /blog from ${blogSource}`);
  });
