# CausalEngine 演示文稿

本目录包含了 CausalEngine™ 的 LaTeX 演示文稿，用于通俗易懂地介绍我们的因果推理引擎。

## 文件说明

- `causal_engine_presentation.tex` - 主演示文稿文件（LaTeX Beamer 格式）
- `causal_engine_presentation.pdf` - 编译生成的演示文稿（20页，英文版本）
- `compile.sh` - 编译脚本
- `generate_figures.py` - 生成演示图表的 Python 脚本
- `README.md` - 本说明文件

## 演示文稿内容

演示文稿包含以下部分：

1. **引言和动机** - 解释为什么需要因果推理
2. **四阶段架构** - 详细介绍 Perception → Abduction → Action → Decision 流程
3. **五种推理模式** - 解释不同的推理模式及其适用场景
4. **性能对比** - 展示在噪声环境下的卓越表现
5. **使用示例** - 快速上手代码示例
6. **理论基础** - 个体选择变量 U 的双重身份
7. **应用场景** - 适合和不适合的使用场景
8. **总结与展望** - 核心贡献和未来方向

## 编译方法

### 前置要求

1. 安装 TeX Live 或 MacTeX（支持中文的 XeLaTeX）
2. 确保系统支持中文字体

### 编译步骤

```bash
# 1. 进入 slide 目录
cd docs/slide

# 2. （可选）生成图表
python3 generate_figures.py

# 3. 编译演示文稿
./compile.sh

# 或手动编译
xelatex causal_engine_presentation.tex
xelatex causal_engine_presentation.tex  # 运行两次以生成正确的目录
```

## 图表说明

演示文稿中引用了以下图表：

1. `cauchy_vs_normal.png` - 柯西分布与正态分布的对比
2. `regression_robustness.png` - 回归任务的噪声鲁棒性对比
3. `classification_robustness.png` - 分类任务的噪声鲁棒性对比
4. `qr_code.png` - 项目主页二维码

如果这些图片不存在，可以：
- 运行 `generate_figures.py` 生成占位图
- 或从项目的 `results/` 目录复制真实的实验结果图

## 自定义修改

### 修改主题

在 `causal_engine_presentation.tex` 中修改：
```latex
\usetheme{Madrid}  % 可改为 Berlin, Warsaw, Copenhagen 等
\usecolortheme{seahorse}  % 可改为 beaver, crane, dolphin 等
```

### 添加新幻灯片

在适当的 section 中添加：
```latex
\begin{frame}{幻灯片标题}
    % 幻灯片内容
\end{frame}
```

### 修改配色

可以自定义颜色主题或单独设置颜色。

## 演示建议

1. **时间控制**：整个演示约需 20-30 分钟
2. **重点突出**：
   - 强调噪声鲁棒性（第4部分）
   - 解释五种推理模式的区别（第3部分）
   - 展示简单的代码示例（第5部分）
3. **互动环节**：在性能对比和应用场景部分可以与听众互动

## 常见问题

1. **中文显示问题**：确保使用 XeLaTeX 而不是 PDFLaTeX
2. **图片缺失**：运行 `generate_figures.py` 或使用真实实验结果
3. **编译错误**：检查是否安装了完整的 TeX Live 发行版

## 导出为其他格式

```bash
# 导出为 PDF（默认）
xelatex causal_engine_presentation.tex

# 导出为讲义模式（每页多个幻灯片）
# 在 tex 文件开头添加 handout 选项：
# \documentclass[aspectratio=169,10pt,handout]{beamer}
```

## 更多资源

- Beamer 用户指南：https://ctan.org/pkg/beamer
- LaTeX 中文支持：https://github.com/CTeX-org/ctex-kit
- 项目主页：https://github.com/1587causalai/causal-sklearn