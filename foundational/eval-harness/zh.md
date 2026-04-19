# LLM 评测框架（lm-evaluation-harness）

- **主要项目**：EleutherAI `lm-evaluation-harness`
- **维护方**：EleutherAI；主要贡献者包括 HuggingFace、Allen AI
- **链接**：[GitHub](https://github.com/EleutherAI/lm-evaluation-harness) · [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) · [论文 arXiv:2110.08207](https://arxiv.org/abs/2110.08207)

## 一句话总结

`lm-evaluation-harness` 是开源 LLM 标准化评测的事实标准框架。它在统一接口下实现了 60 余个基准测试任务——MMLU、HellaSwag、ARC、WinoGrande、GSM8K、HumanEval、TruthfulQA 等——并驱动着 HuggingFace 的 Open LLM Leaderboard。框架将评测分为两种模式：对数似然打分和生成打分，并通过 YAML 配置文件在任务级别进行管理。掌握其计算成本模型对任何认真做评测的工程师都是必修课：在不借助高效推理后端的情况下，对 70B 模型跑完 5-shot MMLU 全量评测需要消耗数百 GPU 小时。

## 背景与动机

`lm-evaluation-harness` 出现之前，基准测试的复现存在严重的不一致性。各个实验室使用各自的评测脚本，分词器、提示词格式、few-shot 样本选取策略各不相同。同一个基准，不同实现可能给出截然不同的分数。该框架通过将每个基准的提示词模板、答案抽取逻辑和指标计算全部编码进版本受控的共享代码库，彻底解决了这个问题。Open LLM Leaderboard 随后将这套框架均匀地应用于所有提交的检查点，使得开放研究社区中跨模型的横向比较第一次具备了真正的意义。

## 两种打分模式

`lm-evaluation-harness` 的核心设计决策是将评测分为对数似然打分和生成打分两种模式。这两种模式不可互换——任务的答案结构决定了只能使用哪种模式。

### 对数似然打分

用于多项选择题任务：MMLU、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、PIQA。

模型从不被要求生成文本。框架对每个候选答案计算模型在给定上下文的条件下赋予该答案的对数概率：

```
score(choice_i) = log P(choice_i | context)
               = sum_{t} log P(token_t | context, choice_i[:<t])
```

预测答案为 `argmax_i score(choice_i)`。对于 4 选 1 的题目，每道题恰好需要 4 次前向传播——每个选项一次。无采样、无温度系数、无束搜索，仅使用 teacher forcing 做一次前向传播。

内存后果：每个选项的前向传播处理 `len(context) + len(choice)` 个 token。在 MMLU 5-shot 场景下，上下文长度为 500–800 个 token，选项长度为 1–10 个 token。计算量的主体集中在共享的上下文前缀，因此前缀缓存（prefix caching）在这里极为有效：上下文 KV 缓存只需计算一次，之后对 4 个选项全部复用。

延迟：在 H100 上对 70B 模型，单次前向传播约 100ms。MMLU 5-shot 共需 56,000 次前向传播（14,079 道题 × 4 个选项），不做批处理的情况下约需 93 分钟。采用高效批处理（4 个选项共享上下文，填充后合并为单个批次），挂钟时间可降至 20–30 分钟。

### 生成打分

用于开放式任务：GSM8K、HumanEval、TruthfulQA-gen、LAMBADA。

模型以标准解码逻辑逐 token 生成完整响应。框架随后对生成输出应用答案抽取函数——精确字符串匹配、正则匹配或代码执行——来判断答案是否正确。

内存后果：生成是自回归的。每个输出 token 都需要对 `prompt_len + 已生成 token 数` 个 token 做一次前向传播，KV 缓存随之递增增长。以 256 个输出 token 为上限，一道 GSM8K 题目在初始 prefill 之后还需要至多 256 次顺序前向传播。GSM8K 全部样本生成的 token 总量：8,500 道题 × 256 token = 217.6 万 token。在 H100 上用 vLLM 跑 70B 模型，持续吞吐约 5,000 token/秒，整个 GSM8K 大约 7 分钟完成。相比对数似然模式，生成模式每个任务的速度更快，因为瓶颈在吞吐量而非前向传播次数。

## 任务定义 Schema

每个任务由一个 YAML 配置文件加可选的 Python 类（用于复杂预处理）共同定义。核心字段如下：

```yaml
task: mmlu_abstract_algebra
dataset_path: hf://datasets/cais/mmlu
dataset_name: abstract_algebra
doc_to_text: "{{question}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: acc
    aggregation: mean
num_fewshot: 5
fewshot_split: validation
output_type: multiple_choice
```

`doc_to_text` 是渲染提示词的 Jinja2 模板；`doc_to_target` 抽取金标签；`output_type: multiple_choice` 将任务路由至对数似然打分路径。生成任务使用 `output_type: generate_until`，并配置 `until` 停止序列列表。

注册一个新任务对大多数基准而言只是编写 YAML 的工作。对于需要自定义预处理的任务（例如 HumanEval 的代码执行环境搭建），Python 类钩子同样可用。这种 schema 设计意味着任务定义是可审计的——YAML 文件的一个 diff 就是对某个基准评测方式的一个 diff。

## Few-Shot 评测

`--num_fewshot N` 参数在每道测试题之前拼接 N 个上下文示例。示例从任务指定的 few-shot 来源（通常是训练集或验证集，与测试集独立）中抽取，并遵循相同的 `doc_to_text` 模板，以换行符作为默认分隔符拼接。

5-shot MMLU 的提示词结构：

```
[示例 1 题目 + 选项 + "Answer: A"]
[示例 2 题目 + 选项 + "Answer: C"]
...
[示例 5 题目 + 选项 + "Answer: B"]
[测试题目 + 选项 + "Answer:"]
```

5-shot MMLU 下，每条提示词约为 500–800 个 token，具体取决于学科（医学题目较长，抽象代数题目较短）。这必须能放入模型的上下文窗口内。对于 4k 上下文窗口的模型，5-shot 已比较紧张；8k 以上则没有问题。

上下文溢出：若 few-shot 提示词超过模型的最大上下文长度，框架默认从左侧截断。这对短上下文模型会悄无声息地损害准确率——测试题目被保留，但部分 few-shot 示例可能被丢弃。

## MMLU 深度解析

MMLU（大规模多任务语言理解）是 LLM 评测中引用最广泛的基准：

- 57 个学术学科：从高中数学到专业医学、法律和伦理学
- 测试集共约 14,079 道题，每个学科大约 100–300 道
- 全部为 4 选 1 多项选择题
- 对数似然打分，无需生成

评测流程：对每道测试题，分别计算 `log P(A | context)`、`log P(B | context)`、`log P(C | context)`、`log P(D | context)`。预测答案取 argmax，与金标签对比。分学科统计准确率，再对 57 个学科做宏平均作为最终分数。

70B 模型 5-shot 的计算量：
- 14,079 道题 × 4 个选项 = 56,316 次前向传播
- 约 100ms/次（H100，70B，提示词约 700 token，批大小 1）= 约 93 分钟
- 批大小 32（每批 128 个选项）：约 3 分钟

批处理带来的 30 倍加速，解释了为何 Open LLM Leaderboard 使用 vLLM 而非朴素的 HuggingFace `generate()` 作为后端。

学科间方差不可忽视：模型在高中科目上通常得分 80% 以上，但在专业法律或道德推理上可能低于 50%。只报告总体平均分会掩盖这种差异。

## GSM8K

GSM8K（小学数学 8K）是标准的算术推理基准：

- 测试集包含 8,500 道小学数学应用题
- 开放式生成任务，无固定候选答案
- 通过正则表达式抽取答案：取响应中最后一个数字
- 与金标准数值做精确匹配

从工程角度看，常见失败模式很能说明问题：模型生成了正确的推理步骤，但给出了错误的最终数字。正则表达式抽取器取响应中最后一个整数或小数，因此输出"答案是 42"时得分，但输出"还剩 42 只鸡，所以农场主有 15 个鸡蛋"时则会被打分为 15 而非 42。这种抽取脆弱性催生了更复杂的答案解析器，但"取最后一个数字"的精确匹配约定已成为可比性的行业标准。

256 token 上限、贪心解码下的计算量：
- 8,500 道题 × 最多 256 token = 最多 217.6 万 token
- 实际上正确答案通常在 200 token 以内，平均生成约 150 token
- H100 + vLLM + 70B，持续 5,000 token/秒：约 7 分钟
- 内存：每条序列的 KV 缓存峰值达 prompt 长度 + 256 token

链式思维提示（CoT）是 GSM8K 的标准做法：`--num_fewshot 8`，使用展示逐步推理的示例。不使用 CoT 时，即使大型模型的分数也会大幅下降。

## HumanEval

HumanEval 衡量 Python 编程的功能正确性：

- 164 道编程题，包含函数签名和文档字符串
- 模型需生成函数体
- 通过对私有测试套件运行评分；所有测试用例通过则视为正确
- 以 pass@k 指标报告结果

pass@k 指标通过多次采样来应对代码生成的随机性：

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

其中 `n` 为每道题的总采样数，`c` 为通过测试的采样数，`k` 为允许提交的次数。以 n=20、pass@1 为例：对每道题生成 20 个完成，统计至少 1 个通过的题目比例，由此估计单次采样通过的概率。这个组合公式提供了无偏估计，无需恰好采样 k 次。

计算量：n=20 采样，164 道题，每个完成约 200 token = 164 × 20 × 200 = 65.6 万 token。以 5,000 token/秒：不到 3 分钟。瓶颈在代码执行而非生成——对 3,280 个测试用例运行可能含危险代码的程序，需要沙箱化执行环境（子进程加超时、Docker 或 Firecracker 等 microVM）。

## Open LLM Leaderboard 基础设施

HuggingFace Open LLM Leaderboard 使用以下基础设施配置对提交的模型检查点运行 `lm-evaluation-harness`：

- 硬件：大多数评测使用单张 A100-80GB；需要张量并行的超大模型使用多卡
- 推理后端：vLLM 提供吞吐量；对 vLLM 不支持的模型回退到原生 HuggingFace `transformers`
- 评测任务（v1 版本）：MMLU（5-shot）、ARC Challenge（25-shot）、HellaSwag（10-shot）、Winogrande（5-shot）、GSM8K（5-shot CoT）、TruthfulQA（0-shot）
- 可复现性：每个模型的结果旁都记录了精确的 harness commit hash 和逐任务配置
- 模型加载：默认 BF16 精度；单卡放不下超大模型时回退 4 位量化
- 吞吐效率：7B 模型跑完全套评测约需 2–3 小时；70B 模型配合 vLLM 约需 12–15 小时

Leaderboard 的价值来自这种可复现性保证。只有用完全相同的方式评测，两个模型的分数才具有可比性。框架通过将所有评测决策编码进版本受控的配置实现了这一点。

## 工程权衡

| 维度 | 方案 A | 方案 B | 核心后果 |
|---|---|---|---|
| 打分模式 | 对数似然（多项选择） | 生成（开放式） | 对数似然：4 次前向传播，无采样，可复用前缀缓存；生成：自回归，单任务更慢，但对无固定答案集的任务不可或缺 |
| 基准类型 | 静态 harness（MMLU、GSM8K） | 实时评测（Chatbot Arena） | 静态：快速可复现，但污染风险随时间增长；实时：最接近用户偏好的真值，积累慢，成本高 |
| Few-shot 数量 | 0-shot | 5-shot | 5-shot 在 MMLU 上提升 5–15 个百分点；每条提示词增加 500–800 token；在短上下文模型上会静默失效 |
| 推理后端 | HuggingFace generate() | vLLM | vLLM 通过 paged attention 带来 10–30 倍吞吐提升；Leaderboard 规模运营的必要条件；增加部署复杂度 |
| 答案抽取 | 正则/精确匹配 | 代码执行（HumanEval） | 代码执行需要沙箱环境；增加基础设施复杂度，但提供功能正确性信号 |
| 评测范围 | 单一基准（GSM8K） | 全套（6 个以上基准） | 全套：70B 模型需 12–15 小时；支持多维对比；计算成本需明确规划 |

## 局限性与基准刷分

静态基准存在根本性的污染问题：随着模型训练数据越来越多地覆盖互联网全量爬取内容，测试集题目会出现在训练数据中。MMLU 的题目被大量复制到博客和学习指南里；GSM8K 的题目出现在数学论坛上。一个在预训练时记住了 MMLU 测试题的模型，汇报的准确率会高估其真实推理能力。

检测难度极大。标准方法——对训练数据与测试集做 n-gram 重叠检测——能发现精确复制，但对改写形式的污染无能为力。部分实验室报告"去污染"结果，但过滤方法参差不齐。

对基础设施工程师的实际影响：高基准分数不再可靠地预测生产质量。一个模型可能在 MMLU 上达到 85%，但在真实用户任务上明显弱于 MMLU 82% 的竞争对手。这正是 Chatbot Arena（见 `../chatbot-arena/`）作为互补信号存在的原因：它持续积累对新提示词的真实人类偏好，这些提示词无法被预先污染。

在生产模型选型中，基准分数应作为筛选器（淘汰明显表现不佳的模型），而非排名标尺（不能假设名次反映生产质量）。最终决策应保留给针对你自身任务分布的内部评测。

## 参考文献

- [1] Gao 等. _A Framework for Few-Shot Language Model Evaluation._ arXiv:2110.08207, 2021.
- [2] Hendrycks 等. _Measuring Massive Multitask Language Understanding._ arXiv:2009.03300, 2020.
- [3] Cobbe 等. _Training Verifiers to Solve Math Word Problems._ arXiv:2110.14168, 2021.（GSM8K）
- [4] Chen 等. _Evaluating Large Language Models Trained on Code._ arXiv:2107.03374, 2021.（HumanEval）
- [5] Clark 等. _Think You Have Solved Question Answering? Try ARC._ arXiv:1803.05457, 2018.
- [6] Zellers 等. _HellaSwag: Can a Machine Really Finish Your Sentence?_ arXiv:1905.07830, 2019.

## 交叉参考

- `../inference-time-scaling/` —— 测试时推理计算扩展；harness 衡量由此产生的输出质量
- `../process-reward-models/` —— PRM 使用在此 harness 内运行的数学基准（GSM8K、MATH）进行评测
- `../../meta/llama3/` —— Llama 3 技术报告引用了由 lm-eval-harness 产出的 MMLU、GSM8K 和 HumanEval 数字
- `../chatbot-arena/` —— 互补的实时评测；基准无法复制的人类偏好信号
