# Personal Homepage

Static personal homepage with:

- bilingual homepage content
- a local web server
- a markdown-to-blog generator under `/blog/`

## Run locally

```powershell
cd C:\Users\zhoujiang\Desktop\personal-homepage
npm.cmd run dev
```

Open:

```text
http://localhost:3000
```

## Notes blog

Put markdown files in:

```text
C:\Users\zhoujiang\Desktop\personal-homepage\notes
```

Then build the blog:

```powershell
npm.cmd run build:blog
```

Generated files will appear in:

```text
C:\Users\zhoujiang\Desktop\personal-homepage\blog
```

You can also point to another notes directory with `BLOG_SOURCE_DIR`.

```powershell
$env:BLOG_SOURCE_DIR='C:\path\to\your\notes'
npm.cmd run build:blog
```
