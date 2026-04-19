# Computer Use

- **Authors / Org**: Anthropic
- **Published**: 2024-10 (public beta with claude-3-5-sonnet-20241022)
- **Links**: [Anthropic blog post](https://www.anthropic.com/news/developing-computer-use) · [API docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) · [reference implementation](https://github.com/anthropic/anthropic-quickstarts/tree/main/computer-use-demo)

## TL;DR

Computer Use is Anthropic's API feature that gives Claude a **screenshot-based action space** over a live desktop or browser environment. The model receives a pixel-rendered screenshot, reasons about what is visible, chooses an action (click, type, scroll, key), executes it, receives the next screenshot, and repeats. There is no DOM access, no accessibility tree, no structured page representation — only pixel-level perception. This universality is both the defining capability and the primary engineering cost: every step incurs a full vision-model inference call, making latency and token budget the dominant constraints for any production deployment.

## Context & Motivation

Browser automation has existed for decades, but all prior approaches require structured access to the application under control:

- **Playwright / Puppeteer / Selenium** — programmatic DOM manipulation; require a controllable browser and structured selectors.
- **RPA tools (UiPath, Automation Anywhere)** — click-coordinate recording; brittle against UI changes, require explicit macro scripting.
- **Browser extensions** — inject JavaScript into the page; limited to the web layer, cannot touch native desktop applications.
- **Accessibility APIs (Windows UIAutomation, macOS AXUIElement)** — expose semantic element trees; application-specific and often incomplete.

Every structured approach fails in the same class of scenarios: legacy desktop applications, PDF viewers, video streams, games, third-party SaaS tools that expose no programmatic API. Computer Use's pixel-only interface works on all of these because it requires nothing from the application itself. The model's job is identical to a human's: look at the screen, decide what to do, move the cursor and press keys.

The capability became viable with the maturation of large vision-language models (VLMs). The same architectural advances that let Claude 3.5 Sonnet describe an image accurately enough also let it ground textual intentions onto pixel coordinates in a screenshot — the model implicitly performs layout parsing, OCR, and spatial reasoning within a single forward pass.

## Action Space

The `computer_use` tool is declared as a tool in the standard Claude API tool-use format. When included in a request, the model can emit it as a tool call in its response. The tool has a single dispatch field, `action`, that selects one of eight sub-actions:

```
screenshot                        → capture current screen state (returns base64 PNG)
left_click(coordinate: [x, y])    → primary click at absolute pixel coordinates
right_click(coordinate: [x, y])   → context menu click
double_click(coordinate: [x, y])  → open/activate element
mouse_move(coordinate: [x, y])    → hover without clicking (for tooltips, hover menus)
type(text: str)                   → keyboard text input at current cursor position
key(text: str)                    → keystroke or chord e.g. "ctrl+c", "Return", "Tab"
scroll(coordinate: [x, y],
       direction: "up"|"down"|"left"|"right",
       amount: int)               → scroll wheel event
```

Coordinates are **absolute pixels** in the screenshot's resolution. The model must infer these from visual inspection of the returned screenshot — it has no ground-truth element positions, no bounding box metadata, no hover events. This is why visual grounding is the hardest subproblem.

The loop is always:

1. Call `screenshot` to get current state.
2. Reason about what to do next (plan + locate target visually).
3. Emit a click/type/scroll/key action.
4. Return to step 1.

The model cannot batch multiple actions atomically. Each action goes to the host, is executed, and the next screenshot reflects the result. This sequential structure means every planning mistake is visible exactly one step later and can be corrected — but it also means that N steps require N round trips.

## Visual Grounding

Visual grounding is the translation of a semantic intention ("click the Submit button") into absolute pixel coordinates. It requires:

1. **Text recognition (OCR)** — find the word "Submit" in the screenshot.
2. **Spatial layout parsing** — determine the bounding box of the button containing that text.
3. **Centroid estimation** — pick a click target at or near the center of that bounding box.

Claude does all of this implicitly via its vision encoder (a ViT-style patch tokenizer followed by cross-attention into the language model backbone — see [VLM Serving](../../multimodal/vlm-serving/)). The model produces `coordinate: [x, y]` directly in its tool call without any intermediate bounding box annotation. When the model is wrong, it either misidentifies the element (clicks the wrong button) or mis-estimates the centroid (clicks near but not on the target). Both failure modes are recoverable in the next step if the model observes that nothing happened or the wrong thing happened.

Empirically, accuracy degrades with:

- **Dense UIs** — many small, closely spaced elements (spreadsheet cells, code editors with gutter controls).
- **Non-standard visual languages** — legacy apps using custom widget toolkits (Qt, Java Swing, old Win32) that don't look like typical web UI.
- **Small text at low resolution** — fonts rendered at 9–11pt in a 720p screenshot become a few pixels wide; OCR accuracy drops sharply.
- **Dynamic content** — loading spinners, animated transitions between screenshots mean the UI in screenshot N is not what the model planned on from screenshot N-1.

Higher resolution screenshots improve accuracy for small elements but increase encoding cost directly. Anthropic's guidance is to scale screenshots to fit within **1366 × 768** before sending — this empirically balances accuracy against token cost for most desktop and browser UIs.

## Latency Arithmetic

A single Computer Use step has the following latency budget:

```
step_latency = t_screenshot + t_encode + t_llm + t_execute

where:
  t_screenshot ≈ 50–150 ms    (X11 framebuffer capture or headless browser screenshot)
  t_encode     ≈ 10–30 ms     (JPEG/PNG compression + base64, on CPU)
  t_llm        ≈ 2,000–5,000 ms  (claude-3-5-sonnet vision inference, network included)
  t_execute    ≈ 100–500 ms   (xdotool click/type, Playwright action, or similar)

typical total: 2.5 – 6 s per step
```

For a 10-step task this gives **25–60 seconds of wall-clock time**. A human completing the same task typically takes 5–15 seconds total. The 3–10x gap is almost entirely the LLM inference time — everything else is sub-second. The implication for task design is that Computer Use is appropriate for tasks where **correctness matters more than latency**: data extraction from a legacy app, form-filling workflows that run overnight, integration with software that has no API. It is not appropriate for interactive, sub-second-response-time workflows.

TTFT (time to first token) for the vision model is also higher than for text-only models because the vision encoder must process the image before the autoregressive generation starts. At 1280 × 800, a Claude vision call has a TTFT approximately 20–40% higher than a text-only call of the same output token count.

## Token Cost

Claude's vision encoding tiles a screenshot into fixed-size patches. A 1280 × 800 image at the standard tiling resolution consumes approximately **1,600–2,000 input tokens** per screenshot. The exact count depends on tiling granularity and whether the image is grayscale or color.

For a 20-step trajectory, the images alone cost:

```
20 screenshots × 1,800 tokens/screenshot = 36,000 tokens
```

Add system prompt (~500 tokens), task description (~200 tokens), and the growing text dialogue (~50 tokens per step × 20 steps = 1,000 tokens), and a 20-step task total context is on the order of **38,000–42,000 tokens**. At Claude 3.5 Sonnet's pricing, this is roughly $0.12–$0.15 per task in input tokens alone.

The critical mitigation is **prefix caching** (see [../building-effective-agents/](../building-effective-agents/)). Screenshots accumulate as a long prefix across the trajectory. If the orchestrator pins each screenshot to the cache as soon as it is received, subsequent steps only incur the cost of computing attention over the new screenshot and new text, not re-encoding all prior screenshots. This can reduce per-step cost by 50–70% for long trajectories. The system prompt and static task instructions should always be in the cache prefix.

For high-frequency deployments (hundreds of tasks per day), cost optimization often means:
- Reduce screenshot frequency — only take a screenshot immediately before a planning step, not after every action.
- Downscale screenshots aggressively — 1024 × 640 is often sufficient for typical web UIs.
- Use prefix caching on the system prompt and long-lived task context.
- Terminate the session as soon as success is detected rather than letting the model take a confirmatory screenshot.

## Resolution Tradeoffs

The relationship between resolution and accuracy follows a roughly logarithmic curve: doubling resolution from 640 × 400 to 1280 × 800 gives a meaningful accuracy improvement; doubling again to 2560 × 1600 gives diminishing returns while quadrupling token count. The practical tradeoff space:

| Resolution | Tokens (approx) | Small-element accuracy | Best for |
|---|---|---|---|
| 800 × 600 | ~900 | Low | High-contrast, large-element UIs |
| 1024 × 768 | ~1,200 | Medium | General web browsing |
| 1280 × 800 | ~1,800 | Good | Recommended default |
| 1920 × 1080 | ~3,600 | High | Dense UIs, spreadsheets, IDEs |
| 2560 × 1440 | ~6,400 | Highest | Rarely necessary, high cost |

Anthropic's recommendation to cap at 1366 × 768 reflects this: above this resolution, cost grows faster than accuracy for most tasks.

## Deployment Infrastructure

A production Computer Use deployment requires:

**Virtual display layer**: A headless environment to render the screen. For Linux, `Xvfb` (X Virtual Framebuffer) provides a virtual X11 display. For browser-only tasks, a headless Chromium instance (via Playwright) is sufficient. For full desktop tasks, a lightweight window manager (Openbox, Fluxbox) on top of Xvfb is standard.

**Screenshot capture**: `scrot`, `import` (ImageMagick), or Playwright's `page.screenshot()`. Latency should be under 100ms; for Xvfb + scrot on a modern server, 50–80ms is typical.

**Action execution**: `xdotool` for keyboard/mouse on X11; Playwright's action API for browser-native tasks. xdotool adds ~20ms of execution overhead; Playwright's browser-side actions are typically faster.

**Session state management**: The model cannot introspect its own action history beyond what is visible in the screenshots. The orchestrator must maintain session state: task description, step count, history of actions taken, success criteria, timeout limits. On unexpected state (e.g., a dialog box the model didn't anticipate), the orchestrator should detect the anomaly and either inject corrective context or trigger a retry.

**Container isolation**: All of this runs in a dedicated Docker container with no network access except the target application. The container is destroyed after each task. This is not optional — Computer Use can execute arbitrary keyboard and mouse actions, including opening terminals, running shell commands, and exiting the target application. Without isolation, a malfunctioning or adversarially-prompted model can do significant damage to the host system.

**Retry logic**: A task-level retry limit (e.g., 3 attempts) and a per-step watchdog timer (e.g., 30 seconds per step) prevent infinite loops. When a step produces no visible change in the UI, the orchestrator can either prompt the model to reconsider or terminate and retry.

## Safety Considerations

Computer Use is one of the highest-risk Claude capabilities from an operational safety standpoint. The model can:

- Open a terminal and execute shell commands.
- Browse to arbitrary URLs (exfiltration of sensitive data visible on screen).
- Delete files via the application's UI.
- Submit forms, make purchases, send emails.

Anthropic's safety guidance for deployments:

1. **Minimal-privilege container**: no internet access except to the target service; no writable host filesystem mounts; no credential forwarding from the host.
2. **Human review checkpoints**: for high-stakes actions (confirm purchase, send email, delete file), the agent workflow should pause and require explicit human approval before proceeding.
3. **Prompt injection vigilance**: an adversary can plant text on-screen (e.g., in a web page being browsed) that instructs the model to take unintended actions. The system prompt should explicitly tell the model to ignore instructions that appear on-screen and that differ from the original task.
4. **Scope restriction in the system prompt**: instruct the model to refuse actions that are out of scope for the task (e.g., "you are filling out a single expense report; do not navigate to any other page").
5. **Rate limiting on destructive actions**: a policy layer between action execution and xdotool/Playwright can reject actions that match destructive patterns — e.g., any keyboard sequence that sends `rm -rf` to a terminal.

## Comparison to Alternative Approaches

| Approach | Accuracy | Setup Complexity | Universality | Latency (per action) | Cost |
|---|---|---|---|---|---|
| Computer Use (vision-based) | Medium–High | Low (containerize + API) | Universal — any UI | 3–6 s (LLM call) | High (vision tokens) |
| Playwright DOM automation | Very High | Medium (selector maintenance) | Web only | 10–100 ms | Low |
| Browser extension (JS injection) | High | Medium | Web only, no native | 10–50 ms | Low |
| RPA (UiPath / Automation Anywhere) | Medium | High (coordinate recording) | Desktop + web | 100–500 ms | Medium (license) |
| Accessibility API (UIAutomation) | High | High (app-specific) | Native apps only | 50–200 ms | Low |
| MCP browser tools | High | Low (protocol layer) | Web only | 50–500 ms | Low |

The decision rule in practice: if the target application is a modern web app with a stable DOM, use Playwright. If the target is a legacy or third-party application with no programmatic interface, use Computer Use. If the target is a mix of both in the same workflow, Computer Use provides a single unified interface at the cost of latency and per-step LLM inference budget.

## Relation to Agent Patterns

Computer Use is the "augmented LLM" pattern from [../building-effective-agents/](../building-effective-agents/) taken to its logical extreme: the LLM's tools are not structured API calls but raw UI primitives. The agent loop is the same perception-reasoning-action cycle, but perception is a pixel array and action is a mouse coordinate.

This means all agent-loop engineering considerations apply at higher intensity:

- **Context window management** is critical because screenshots are large tokens that accumulate fast.
- **Error recovery** must be robust because UI states are more varied than structured API responses.
- **Parallelism** is limited — steps are inherently sequential (you can't click two things simultaneously), though multiple independent task sessions can run in parallel across containers.

Computer Use tasks that span multiple distinct application contexts (e.g., copy data from App A, paste into App B, save to App C) are best modeled as multi-agent workflows where each sub-agent has a single-application scope and a parent orchestrator passes state between them — reducing the context window per agent and confining failure domains.

MCP (see [../mcp/](../mcp/)) can expose Computer Use actions as MCP tools, allowing a host application to delegate UI control to a Computer Use subprocess server. This is useful when one larger agentic system needs UI interaction as one capability among many, without entangling the vision-model infrastructure with the main orchestration layer.

## Cross-References

- [../building-effective-agents/](../building-effective-agents/) — orchestration patterns for multi-step agent workflows; prompt caching and context management in long sessions
- [../../multimodal/vlm-serving/](../../multimodal/vlm-serving/) — the serving infrastructure for vision-language models that processes screenshots
- [../../multimodal/llava/](../../multimodal/llava/) — how visual tokens are constructed and how they consume context window space
- [../mcp/](../mcp/) — MCP can surface Computer Use actions as protocol-standard tools for any MCP-compatible host
