# Personal Homepage

Static personal homepage with:

- bilingual homepage content
- a printable HTML CV under `/cv/`
- a local web server
- a local full-notes blog preview and a public-only blog export pipeline

## Run locally

```bash
cd /Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage
npm run dev
```

Open:

```text
http://localhost:3000
```

## Open In Obsidian

这个项目现在已经带有一个本地 Obsidian vault 配置。
打开这个 vault 时，会默认进入 [BLOG.md](/Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage/BLOG.md)，并在页面里内嵌本地 blog 界面。

直接打开：

```bash
cd /Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage
npm run open:obsidian-blog
```

如果 iframe 没内容，先启动本地站点：

```bash
cd /Users/jiangzhou/Documents/personal-dev-workspace
npm run dev:site
```

## Notes blog

Put markdown files in:

```text
/Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage/notes
```

Then build the blog:

```bash
npm run build:blog
```

Generated files will appear in:

```text
/Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage/.local/blog
```

You can also point to another notes directory with `BLOG_SOURCE_DIR`.

```bash
BLOG_SOURCE_DIR="/path/to/your/notes" npm run build:blog
```

`npm run dev` 和 `npm start` 现在都会先构建这份本地 blog，再启动站点。
本地访问 `http://localhost:3000/blog/index.html` 时，server 会优先读取 `.local/blog`，因此你看到的是全量笔记版本，包含 `private`。

## Note Visibility

每篇笔记现在都用 front matter `tags` 标注可见性，约定如下：

- `tags: [private]` 表示默认私有；
- `tags: [public]` 表示允许公开发布；
- 如果缺少这两个标签，系统会按 `private` 处理。

要公开一篇笔记，直接把它的 front matter 改成：

```yaml
tags: [public]
```

给现有笔记补齐默认可见性标签：

```bash
cd /Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage
npm run notes:visibility
```

只检查、不写回：

```bash
npm run notes:visibility:check
```

只构建 public 笔记：

```bash
npm run build:blog:public
```

这条命令会把公开可发布的静态页面写到：

```text
/Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage/blog
```

也就是说：

- `.local/blog`：本地预览专用，默认全量，包含 `private`；
- `blog/`：公开站点产物，只放 `public`，可以安全提交到 GitHub Pages。

## Publish Public Notes

公开发布脚本只会导出 `tags: [public]` 的笔记，并会：

- 只构建 public blog 页面；
- 禁用 blog 模板里对私有图片素材的引用；
- 默认拒绝发布包含本地绝对路径或 `mai-private` 资源引用的 public 笔记；
- 将导出结果推送到你指定的公开仓库。

推荐先在本地准备一个公开仓库的 clone，再执行：

```bash
cd /Users/jiangzhou/Documents/personal-dev-workspace/apps/personal-homepage
npm run publish:public -- --repo /absolute/path/to/public-repo --branch main
```

如果你当前这个仓库本身就是 GitHub Pages 站点仓库，那么日常流程建议改成：

```bash
npm run dev
# 本地检查全量笔记效果（含 private）

npm run build:blog:public
# 刷新仓库里的公开 blog/ 产物

git add blog
git commit -m "Update public notes"
git push
```

如果你想直接从远程仓库 URL clone 后再推送：

```bash
npm run publish:public -- --repo-url git@github.com:your-name/your-public-repo.git --branch main
```

可选参数：

- `--base-path /repo-name`：显式指定 GitHub Pages 的 base path；
- `--message "Publish public notes"`：自定义提交信息；
- `--allow-dirty`：允许目标公开仓库工作区有未提交变更；
- `--allow-empty`：即使当前没有 public 笔记也继续导出；
- `--dry-run`：只生成导出包，不实际提交推送。
