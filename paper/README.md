# Causal Regression 论文项目管理

> **项目概述**: 将CausalEngine理论框架转化为顶级学术论文的完整项目管理文档

## 🎯 项目目标

将已完成的**CausalEngine理论框架和代码实现**转化为高质量学术论文，目标发表于顶级ML会议（ICML, NeurIPS, ICLR）。

### 核心交付物
- **学术论文**: 完整的理论阐述和实验验证
- **开源代码**: 可复现的实验和算法实现
- **技术文档**: 支持社区使用和扩展

## 📁 项目架构

```
paper/
├── README.md                      # 📋 项目总览（本文件）
├── strategy/                      # 🎯 策略规划
│   ├── paper_strategy.md          # 详细的论文策略与规划
│   └── concept_definition.md      # Causal Regression概念定义
├── content/                       # 📝 内容草稿
│   ├── abstract.md                # 摘要草稿
│   ├── introduction.md            # 引言部分
│   ├── methodology.md             # 方法论部分
│   ├── experiments.md             # 实验设计
│   └── conclusion.md              # 结论部分
├── latex/                         # 📄 LaTeX文档（已废弃，保留作参考）
│   ├── main.tex                   # 主论文文件
│   ├── references.bib             # 参考文献库
│   ├── sections/                  # 各章节tex文件
│   └── figures/                   # 图表文件
├── AuthorKit26/                   # 🎯 AAAI 2026 投稿目录
│   └── AnonymousSubmission/
│       └── LaTeX/                 # 📄 正式投稿LaTeX文档
│           ├── anonymous-submission-latex-2026.tex  # 主投稿文件
│           ├── aaai2026.bib       # 参考文献
│           ├── aaai2026.bst       # 引用样式
│           ├── aaai2026.sty       # AAAI样式文件
│           └── figures/           # 图表文件
└── notes/                         # 📝 工作笔记
    ├── meeting_notes.md           # 讨论记录
    └── ideas.md                   # 想法收集
```

## 📚 项目文档结构

### 核心文档分工
- **`paper_strategy.md`**: 📋 完整的论文写作策略、概念定义、学术定位
- **`concept_definition.md`**: 🎯 "Causal Regression"的精确定义和理论基础
- **`AuthorKit26/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.tex`**: 📄 AAAI 2026正式投稿LaTeX主文件
- **`AuthorKit26/AnonymousSubmission/LaTeX/aaai2026.bib`**: 📚 AAAI格式参考文献库

### 写作协作架构
- **`content/`**: 分章节草稿，支持并行写作（作为投稿文件的内容源）
- **`latex/`**: 原始LaTeX文档（已废弃，保留作参考）
- **`AuthorKit26/AnonymousSubmission/LaTeX/`**: 🎯 **正式投稿目录**，AAAI 2026标准格式
- **`notes/`**: 会议记录、想法收集和临时笔记

## 🎯 当前投稿设置

**投稿目录**: `@paper/AuthorKit26/AnonymousSubmission/LaTeX/`

**主要文件**:
- `anonymous-submission-latex-2026.tex`: 主投稿文件
- `aaai2026.bib`: 参考文献库
- `aaai2026.sty`: AAAI 2026样式文件（不可修改）
- `aaai2026.bst`: 引用格式（不可修改）

## ⏰ 项目时间规划

### 第1阶段 (2周): 概念与相关工作
- [ ] 完善概念定义
- [ ] 相关工作梳理
- [ ] Introduction初稿

### 第2阶段 (3周): 理论与算法
- [ ] 数学框架完善
- [ ] CausalEngine算法描述
- [ ] Methodology章节

### 第3阶段 (4周): 实验验证
- [ ] 实验设计与实施
- [ ] 结果分析与可视化
- [ ] Experiments章节

### 第4阶段 (2周): 完善与投稿
- [ ] 全文修改完善
- [ ] 格式调整与检查
- [ ] 投稿准备

## 🎖️ 项目成功标准

### 短期目标
- [ ] 完成高质量论文初稿
- [ ] 通过顶级会议/期刊审稿
- [ ] 获得学术界认可

### 长期影响
- [ ] "Causal Regression"成为认可术语
- [ ] CausalEngine被广泛使用
- [ ] 推动因果AI领域发展
- [ ] 建立研究者社区

## 📈 当前项目状态

### 已完成 ✅
- 理论框架完备（参见 `paper_strategy.md`）
- 代码实现完成（参见 `../causal_sklearn/`）
- AAAI 2026投稿环境设置（参见 `AuthorKit26/AnonymousSubmission/LaTeX/`）
- 概念定义清晰（参见 `strategy/concept_definition.md`）
- 投稿论文初始结构和Abstract/Introduction已完成

### 进行中 🔄
- 内容创作和写作
- 实验设计和数据收集
- 文献综述和相关工作整理

### 下一步行动 📋
1. 开始Introduction章节的详细写作
2. 完善实验设计和数据收集
3. 准备核心图表和可视化
4. 寻求合作者和反馈

## 🤝 协作指南

### 版本控制
- 使用Git进行版本管理
- 分支策略：feature/章节名称
- 提交信息格式：`[章节] 具体修改内容`

### 文档更新
- 重大修改需更新本README
- 策略变更需同步更新 `paper_strategy.md`
- 所有会议和讨论记录在 `notes/meeting_notes.md`

---

**项目愿景**: 通过这篇论文，将CausalEngine的技术突破转化为学术影响力，为AI领域贡献因果智能的新范式。