const toggle = document.querySelector(".theme-toggle");
const langToggle = document.querySelector(".lang-toggle");
const reveals = document.querySelectorAll(".reveal");
const themeStorageKey = "personal-homepage-theme";
const langStorageKey = "personal-homepage-lang";
const pageTitle = document.querySelector("title");
const metaDescription = document.querySelector('meta[name="description"]');
const visitorApiUrl = "https://bsz.saop.cc/api";
const visitorElements = {
  root: document.querySelector("[data-visitor-counter]"),
  note: document.querySelector("[data-visitor-note]"),
  siteUv: document.getElementById("visitor-site-uv"),
  sitePv: document.getElementById("visitor-site-pv"),
  pagePv: document.getElementById("visitor-page-pv")
};
let activeLang = "en";
let visitorState = "loading";
let visitorStats = null;

const translations = {
  en: {
    htmlLang: "en",
    pageTitle: "Zhou Jiang | AI Researcher",
    description:
      "Zhou Jiang is an undergraduate AI researcher working on diffusion models, preference alignment, reinforcement learning, and representation learning.",
    "nav.about": "About",
    "nav.research": "Research",
    "nav.publications": "Publications",
    "nav.blog": "Blog",
    "nav.contact": "Contact",
    "hero.eyebrow": "Undergraduate AI Researcher",
    "hero.name": "Zhou Jiang",
    "hero.subtitle": "Applying for PhD programs in generative modeling and machine learning.",
    "hero.text":
      "I am an undergraduate researcher at South China University of Technology and a visiting student at Westlake University. My current work focuses on diffusion models, preference alignment, reinforcement learning, and representation learning.",
    "hero.primaryCta": "Selected Papers",
    "hero.secondaryCta": "Download CV",
    "focus.title": "Current Focus",
    "focus.item1": "Preference optimization for diffusion models",
    "focus.item2": "Inference-time alignment and guidance",
    "focus.item3": "Foundational theory of generative models",
    "focus.item4": "Connections between generative modeling and representation learning",
    "focus.historyTitle": "Previously Explored",
    "focus.history1": "Offline RL for black-box optimization",
    "focus.history2": "Deep learning on financial data",
    "stats.one.value": "Westlake",
    "stats.one.label": "Visiting student",
    "stats.two.value": "Diffusion",
    "stats.two.label": "Alignment focus",
    "stats.three.value": "arXiv",
    "stats.three.label": "Recent preprint",
    "visitors.title": "Live Visitor Count",
    "visitors.siteUv": "Visitors",
    "visitors.sitePv": "Site Views",
    "visitors.pagePv": "Page Views",
    "visitors.noteLoading": "Loading live stats...",
    "visitors.noteLive": "Synced automatically for the public site.",
    "visitors.noteLocal": "Local preview does not increment the public counter.",
    "visitors.noteError": "Visitor counter is temporarily unavailable.",
    "visual.heroAlt": "Sakurajima Mai inspired illustration in warm dusk light",
    "about.eyebrow": "About",
    "about.title": "Research-driven training with enough engineering depth to turn ideas into reproducible systems.",
    "about.p1":
      "I am pursuing a dual degree in Computer Science and Technology plus Finance at South China University of Technology. My goal is to pursue a PhD in machine learning, especially in generative models and alignment.",
    "about.p2":
      "I care about questions that connect principled learning objectives with practical model behavior. In recent work, I have built end-to-end pipelines for preference data processing, diffusion finetuning, evaluation, and inference-time guidance.",
    "research.eyebrow": "Research Experience",
    "research.title": "Projects that show both independent thinking and execution.",
    "research.item1.time": "2025 - Now",
    "research.item1.title": "Diffusion Alignment and Optimization, Westlake University",
    "research.item1.text":
      "Visiting student working with Yandong Wen and Zhen Liu on preference optimization for diffusion models. Built an end-to-end pipeline covering preference data, finetuning, plug-and-play guidance, and benchmark evaluation. The resulting work is currently available as an arXiv preprint.",
    "research.item2.time": "2024",
    "research.item2.title": "MetaBBO via Offline Q-value Learning, SCUT GMC Lab",
    "research.item2.text":
      "Worked with Yuejiao Gong on reinforcement-learning-based dynamic algorithm configuration for black-box optimization. Implemented most experiments and core code, with emphasis on reproducibility and robust comparison across tasks.",
    "research.item3.time": "Side Work",
    "research.item3.title": "Applied Computer Vision and LLM Projects",
    "research.item3.text":
      "Built systems for handwritten formula recognition, text eradication from document images, and lightweight finetuning of a 1.5B Qwen model for finance-oriented QA and analysis.",
    "publications.eyebrow": "Selected Publications",
    "publications.title": "Evidence that the research story is already forming before PhD applications.",
    "publications.paper1.tag": "Preprint",
    "publications.paper1.title":
      "Rethinking Preference Alignment for Diffusion Models with Classifier-Free Guidance",
    "publications.paper1.text":
      "Zhou Jiang, Yandong Wen, Zhen Liu. A study of preference optimization for diffusion models, with a principled connection between classifier-free guidance and preference learning.",
    "publications.paper1.link1": "arXiv",
    "publications.more": "More Publications",
    "publications.paper2.tag": "Co-author",
    "publications.paper2.text":
      "ICML 2025 paper on offline reinforcement learning for meta black-box optimization, dynamic algorithm configuration, and Q-function decomposition.",
    "publications.paper2.link1": "Paper",
    "publications.paper2.link2": "Code",
    "publications.paper3.tag": "Co-author",
    "publications.paper3.text":
      "IEEE TGRS paper on multi-scale remote sensing object detection with a serial pyramid convolutional design.",
    "publications.paper3.link1": "DOI",
    "publications.paper4.tag": "Co-author",
    "publications.paper4.text":
      "JVCIR paper on pseudo-label refinement for weakly supervised semantic segmentation by integrating saliency cues.",
    "publications.paper4.link1": "DOI",
    "publications.paper4.link2": "Code",
    "extras.eyebrow": "Awards and Skills",
    "extras.title": "Strong signals beyond publication count.",
    "extras.p1":
      "Honors include Provincial First Prize in the China Undergraduate Mathematics Competition, International Second Prize in MCM/ICM in both 2023 and 2024, and National First Prize in the HuaShuCup Mathematical Modeling Competition.",
    "extras.p2":
      "I mainly work with Python, PyTorch, PyTorch Lightning, C/C++, MATLAB, SQL, and JavaScript/TypeScript. I value reproducible experimentation, clean evaluation pipelines, and clear technical communication.",
    "contact.eyebrow": "Contact",
    "contact.title":
      "If your group is working on generative modeling or alignment, I would be glad to connect.",
    "contact.cv": "Curriculum Vitae",
    "contact.blog": "Notes Blog",
    "contact.links1": "South China University of Technology",
    "contact.links2": "Westlake University",
    "contact.links3": "Latest arXiv Preprint",
    themeToggle: "Toggle Theme",
    langToggle: "中文"
  },
  zh: {
    htmlLang: "zh-CN",
    pageTitle: "姜洲 | 人工智能研究者",
    description:
      "姜洲是一名本科 AI 研究者，研究方向包括扩散模型、偏好对齐、强化学习与表征学习。",
    "nav.about": "简介",
    "nav.research": "研究经历",
    "nav.publications": "论文",
    "nav.blog": "博客",
    "nav.contact": "联系",
    "hero.eyebrow": "本科 AI 研究者",
    "hero.name": "姜洲",
    "hero.subtitle": "正在申请生成式建模与机器学习方向的 PhD 项目。",
    "hero.text":
      "我目前就读于华南理工大学，是一名本科研究者，并在西湖大学进行访问研究。当前主要关注扩散模型、偏好对齐、强化学习和表征学习。",
    "hero.primaryCta": "查看论文",
    "hero.secondaryCta": "下载简历",
    "focus.title": "当前关注",
    "focus.item1": "扩散模型的偏好优化",
    "focus.item2": "推理阶段的对齐与引导",
    "focus.item3": "生成模型的基础理论",
    "focus.item4": "生成建模与表征学习之间的联系",
    "focus.historyTitle": "过往关注",
    "focus.history1": "面向黑盒优化的离线强化学习",
    "focus.history2": "金融数据上的深度学习",
    "stats.one.value": "西湖大学",
    "stats.one.label": "访问学生",
    "stats.two.value": "扩散模型",
    "stats.two.label": "对齐研究",
    "stats.three.value": "arXiv",
    "stats.three.label": "近期预印本",
    "visitors.title": "实时访客统计",
    "visitors.siteUv": "访客人数",
    "visitors.sitePv": "站点访问",
    "visitors.pagePv": "本页访问",
    "visitors.noteLoading": "正在加载访问统计...",
    "visitors.noteLive": "公开站点会自动同步这些数据。",
    "visitors.noteLocal": "本地预览不会计入公开站点统计。",
    "visitors.noteError": "访客统计暂时不可用。",
    "visual.heroAlt": "带有黄昏光感的樱岛麻衣风格插图",
    "about.eyebrow": "个人简介",
    "about.title": "以研究为主线，同时具备把想法落成可复现系统的工程能力。",
    "about.p1":
      "我目前在华南理工大学攻读计算机科学与技术和金融学双学位，希望继续攻读机器学习方向的 PhD，尤其关注生成模型与对齐问题。",
    "about.p2":
      "我比较关心那些能够把原理性学习目标和实际模型行为连接起来的问题。最近的工作中，我搭建了从偏好数据处理、扩散模型微调到评测与推理引导的完整实验流程。",
    "research.eyebrow": "研究经历",
    "research.title": "这些项目既体现研究判断，也体现独立执行能力。",
    "research.item1.time": "2025 - 至今",
    "research.item1.title": "扩散模型对齐与优化，西湖大学",
    "research.item1.text":
      "在 Yandong Wen 和 Zhen Liu 指导下，以访问学生身份研究扩散模型的偏好优化问题。我搭建了覆盖偏好数据、模型微调、即插即用引导和基准评测的完整流程，相关工作目前以 arXiv 预印本形式公开。",
    "research.item2.time": "2024",
    "research.item2.title": "基于离线 Q-value Learning 的 MetaBBO，华南理工大学 GMC Lab",
    "research.item2.text":
      "与龚悦娇老师合作，研究基于强化学习的黑盒优化动态算法配置方法。我完成了大部分实验和核心代码实现，并重点保证结果的可复现性与任务间对比的稳健性。",
    "research.item3.time": "其他项目",
    "research.item3.title": "计算机视觉与 LLM 应用项目",
    "research.item3.text":
      "做过手写公式识别、文档图像手写内容擦除，以及面向金融问答与分析的 1.5B Qwen 轻量微调项目。",
    "publications.eyebrow": "代表论文",
    "publications.title": "在正式申请 PhD 之前，先把研究主线和论文信号建立起来。",
    "publications.paper1.tag": "预印本",
    "publications.paper1.title": "Rethinking Preference Alignment for Diffusion Models with Classifier-Free Guidance",
    "publications.paper1.text":
      "Zhou Jiang, Yandong Wen, Zhen Liu. 该工作研究扩散模型中的偏好优化问题，并尝试从原理上建立 classifier-free guidance 与偏好学习之间的联系。",
    "publications.paper1.link1": "arXiv",
    "publications.more": "更多论文",
    "publications.paper2.tag": "合作论文",
    "publications.paper2.text":
      "ICML 2025 论文，研究面向元黑盒优化的离线强化学习、动态算法配置和 Q-function 分解。",
    "publications.paper2.link1": "论文",
    "publications.paper2.link2": "代码",
    "publications.paper3.tag": "合作论文",
    "publications.paper3.text":
      "IEEE TGRS 论文，围绕遥感目标检测中的多尺度建模提出串行金字塔卷积设计。",
    "publications.paper3.link1": "DOI",
    "publications.paper4.tag": "合作论文",
    "publications.paper4.text":
      "JVCIR 论文，通过融合显著性线索改进弱监督语义分割中的伪标签生成。",
    "publications.paper4.link1": "DOI",
    "publications.paper4.link2": "代码",
    "extras.eyebrow": "奖项与技能",
    "extras.title": "除了论文数量之外，也保留能体现基础能力的信号。",
    "extras.p1":
      "获得过全国大学生数学竞赛省一等奖、MCM/ICM 国际二等奖，以及华数杯全国一等奖等竞赛奖项。",
    "extras.p2":
      "主要使用 Python、PyTorch、PyTorch Lightning、C/C++、MATLAB、SQL 和 JavaScript/TypeScript。我重视实验可复现性、评测流程设计和清晰的技术表达。",
    "contact.eyebrow": "联系我",
    "contact.title": "如果你的研究组关注生成模型或对齐问题，欢迎联系我。",
    "contact.cv": "个人简历",
    "contact.blog": "笔记博客",
    "contact.links1": "华南理工大学",
    "contact.links2": "西湖大学",
    "contact.links3": "最新 arXiv 预印本",
    themeToggle: "切换配色",
    langToggle: "EN"
  }
};

function getLocale(lang) {
  return translations[lang] || translations.en;
}

function applyTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
}

function applyLanguage(lang) {
  activeLang = lang;
  const locale = getLocale(lang);
  document.documentElement.lang = locale.htmlLang;
  pageTitle.textContent = locale.pageTitle;
  metaDescription.setAttribute("content", locale.description);

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    if (locale[key]) {
      node.textContent = locale[key];
    }
  });

  document.querySelectorAll("[data-i18n-alt]").forEach((node) => {
    const key = node.dataset.i18nAlt;
    if (locale[key]) {
      node.setAttribute("alt", locale[key]);
    }
  });

  toggle.textContent = locale.themeToggle;
  langToggle.textContent = locale.langToggle;
  document.body.classList.toggle("zh", lang === "zh");
  renderVisitorStats();
}

function formatVisitorNumber(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }

  return new Intl.NumberFormat(activeLang === "zh" ? "zh-CN" : "en-US").format(value);
}

function setVisitorValue(node, value) {
  if (!node) {
    return;
  }

  node.textContent = formatVisitorNumber(value);
}

function renderVisitorStats() {
  if (!visitorElements.root) {
    return;
  }

  const locale = getLocale(activeLang);
  visitorElements.root.dataset.state = visitorState;

  if (visitorStats) {
    setVisitorValue(visitorElements.siteUv, visitorStats.siteUv);
    setVisitorValue(visitorElements.sitePv, visitorStats.sitePv);
    setVisitorValue(visitorElements.pagePv, visitorStats.pagePv);
  } else {
    setVisitorValue(visitorElements.siteUv, NaN);
    setVisitorValue(visitorElements.sitePv, NaN);
    setVisitorValue(visitorElements.pagePv, NaN);
  }

  if (!visitorElements.note) {
    return;
  }

  if (visitorState === "ready") {
    visitorElements.note.textContent = locale["visitors.noteLive"];
    return;
  }

  if (visitorState === "local") {
    visitorElements.note.textContent = locale["visitors.noteLocal"];
    return;
  }

  if (visitorState === "error") {
    visitorElements.note.textContent = locale["visitors.noteError"];
    return;
  }

  visitorElements.note.textContent = locale["visitors.noteLoading"];
}

function shouldTrackVisitorStats() {
  if (window.location.protocol === "file:") {
    return false;
  }

  const blockedHosts = new Set(["localhost", "127.0.0.1"]);
  return !blockedHosts.has(window.location.hostname);
}

async function loadVisitorStats() {
  if (!visitorElements.root) {
    return;
  }

  if (!shouldTrackVisitorStats()) {
    visitorState = "local";
    renderVisitorStats();
    return;
  }

  visitorState = "loading";
  renderVisitorStats();

  try {
    const response = await fetch(visitorApiUrl, {
      method: "POST",
      credentials: "include",
      headers: {
        "x-bsz-referer": window.location.href
      }
    });

    if (!response.ok) {
      throw new Error(`Visitor counter request failed with ${response.status}`);
    }

    const payload = await response.json();

    if (!payload || !payload.success || !payload.data) {
      throw new Error("Visitor counter returned an invalid payload");
    }

    visitorStats = {
      siteUv: Number(payload.data.site_uv),
      sitePv: Number(payload.data.site_pv),
      pagePv: Number(payload.data.page_pv)
    };
    visitorState = "ready";
  } catch (error) {
    console.warn("Visitor counter failed to load.", error);
    visitorState = "error";
  }

  renderVisitorStats();
}

const savedTheme = localStorage.getItem(themeStorageKey);
if (savedTheme) {
  applyTheme(savedTheme);
}

const savedLang = localStorage.getItem(langStorageKey) || "en";
applyLanguage(savedLang);

toggle.addEventListener("click", () => {
  const nextTheme = document.body.classList.contains("dark") ? "light" : "dark";
  applyTheme(nextTheme);
  localStorage.setItem(themeStorageKey, nextTheme);
});

langToggle.addEventListener("click", () => {
  const currentLang = localStorage.getItem(langStorageKey) || "en";
  const nextLang = currentLang === "en" ? "zh" : "en";
  applyLanguage(nextLang);
  localStorage.setItem(langStorageKey, nextLang);
});

loadVisitorStats();

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15 }
);

reveals.forEach((item) => observer.observe(item));
