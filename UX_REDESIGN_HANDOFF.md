# UX_REDESIGN_HANDOFF — 苹果风 UX 重构（续做）

> 2026-07-03。给下个 session 接手用。**先读这份 + 记忆里的 [[oslian-ux-redesign]]、[[oslian-osler-project]]、
> [[oslian-data-source-provenance]]。** 文中 file:line / 函数名动手前对现有代码验一遍。

---

## 0. TL;DR —— 现在到哪了

用户觉得 app "太复杂太臃肿、不够简洁人性化"，北极星 = **苹果审美**。已经做完两大块并 commit：
**①砍冗余 + Rx 并入 Reasoning ②Apple 风重做 Analyze + 答案优先药卡**。现在要做**三个打磨项**（见 §3）。

- 分支 **`feat/voice-input`**，工作树干净，tsc 一路过。**领先 `origin` 3 个 commit、未 push**：
  - `aae0265` feat(ux): Apple-native Analyze + answer-first drug cards
  - `f5a7ece` refactor(ux): remove Chat, trim Settings clutter, fold Rx into Reasoning
  - `a5ed741` feat(sources): official-only citation links + openFDA on by default
- **底部现在 4 tab**：Patients · Analyze · Reasoning · Settings（Chat 砍了，Rx 并进 Reasoning 的第 4 段）。
- **视觉规范（可分享）**：`https://claude.ai/code/artifact/05015c07-8732-4786-a2de-c01925889227`（色板/字阶/组件/三屏 mock）。

## 1. 铁律（别破坏）

- **保留 app 现有 `Outfit/Sora` 字体 + 海军蓝 `#0F2B5B`**（用户明确不换 SF、不换 iOS 蓝）。tokens 在 `mobile/src/theme/tokens.ts`。
- **UI 不加装饰性 emoji**（用户嫌"太 AI"；`⚠️` 这类功能告警可留）。
- **所有业务逻辑保留**：语音、`case_parser`/`parseCase`、五项必填门槛（indication/age/sex/weight/eGFR）、样例/深链导入、analyze 调用、Photon 开药——**只改外壳和信息层级**。
- 每步 **tsc + 模拟器**验证。

## 2. 已完成（已 commit，别重做）

- **砍**：Chat 整个 tab（screen+MessageBubble+TraceCard+chat() API+AppContext chat 管线）；Settings 的 BACKEND 调试段 + LLM API-key 段；Analyze 的 openFDA 开关；Rx 里重复的患者切换器。
- **Rx 并入 Reasoning**：新 `components/RxPanel.tsx` 是 Reasoning 第 4 段（`Drugs|Graph|Disease|Rx`）；删了 `app/prescriptions.tsx`；`prescribe.tsx` 签发后跳 `/reasoning?view=rx`。
- **Analyze 苹果风**（`app/analyze.tsx`）：大标题 "New case" + 副标题 → `VoiceDictation`（现在含 textarea + 内联 navy 圆麦，见 `components/VoiceDictation.tsx`）→ "Structure note" 按钮 → CapturedFields → **渐进披露** "Start from an example"（`CasePresetChips`）/"Enter details"（6 字段表单），`showExamples`/`showDetails` state → 海军蓝 "Get recommendations"。
- **答案优先药卡**（`components/DrugCard.tsx`）：默认只显 rank+name+safety badge+`final_answer`(结论大字)+dose(blocked 改一行 `doseNote`)+Prescribe；rationale/warnings/mechanism 收进 **"Why this"** 折叠、`DrugSources` 收进 **"Sources · evidence"** 折叠（`showWhy`/`showSources`）；删了裸 `mechanism_score`。

## 3. ⏭️ 待打磨（本 session 用户点名的三项）

1. **Analyze "Structure note" 改自动解析**：现在打字后要手点 "Structure note" 才出 CAPTURED（语音已经在 `handleTranscript` 里自动 `runParse` 了）。目标：打字停顿/失焦后自动解析、去掉手动按钮。做法建议：给 `VoiceDictation` 的 textarea 加 `onBlur`，或在 `analyze.tsx` 对 `text` 加 debounce effect 调 `runParse(text)`（text 非空且变化时）；然后隐藏/删掉 "Structure note" Pressable。`runParse` 逻辑不动。**注意**：`runParse` 走 `/api/parse_case`——见 §5 后端坑。
2. **Graph / Disease 降级**（`app/reasoning.tsx`）：现在四段 `Drugs|Graph|Disease|Rx` 平级。Graph（vis-network 图谱，手机上难读）、Disease（疾病面板）是低频参考视图。降级思路：主段留 `Drugs`（+ 也许 `Rx`），把 Graph/Disease 收进一个次级入口（如顶部一个 "Reasoning graph / Disease model" 的小链接、或一个 "More ▾"）。`SEGMENTS` 数组 + 分段渲染在 `reasoning.tsx:27-33, 126-205`。**别动** Graph→Drugs 的 highlight/scroll 桥（`GraphWebView` postMessage → `setHighlightDrug`）。
3. **Reasoning 顶部患者条再收一下**（`app/reasoning.tsx` summary strip，~104-123 + styles ~246-288）：indication + 年龄性别 + 红色 flag pills。让它更紧凑/干净（苹果风），别占太多竖向空间。

（用户本 session 明确 **"Structure note" 先保留**过——但现在是要把它改成自动解析，即上面 #1。）

## 4. 怎么跑 / 验证

- **模拟器**：iPhone 17 / iOS 26.5，UDID `AF02103D-2418-45DE-85C5-AED1472E0A54`。app bundleId `com.alphonse1223.oslian-rx`，scheme `oslianrx://`。
- **重载 app**：`xcrun simctl terminate <udid> com.alphonse1223.oslian-rx; xcrun simctl launch <udid> com.alphonse1223.oslian-rx`（JS 改动 Metro 热更，但整重载最稳）。
- **深链自动分析**（不用手点，直达 Reasoning）：`xcrun simctl openurl <udid> "oslianrx://analyze?patient=<urlencoded JSON>&auto=1"`。ACS 测试 payload：`{"title":"64M · ACS","indication":"acute coronary syndrome","age":64,"sex":"M","weight_kg":82,"egfr":72,"symptoms":["crushing chest pain","diaphoresis","dyspnea"],"vitals":{"sbp":88,"heart_rate":112,"spo2":94}}`。
- **tsc**：`cd mobile && npx tsc --noEmit`（先 `eval "$(/opt/homebrew/bin/brew shellenv)"; export LANG=en_US.UTF-8`）。
- **截屏**：`xcrun simctl io <udid> screenshot <path.png>`；**滚动/点击**要用 computer-use（模拟器无 idb）。

## 5. 坑 / 关键事实

- **后端 `/api/parse_case` 会 404**：用户机器上跑的 :5000 后端可能是**旧代码、缺 parse_case 路由** → "Structure note"/语音解析报 `JSON Parse error: Unexpected character: <`（后端返回 404 HTML）。**核心 analyze 不受影响**（`/api/analyze` 在）。修法 = 用户重启后端（`cd Osler; HOST=127.0.0.1 PORT=5000 .venv/bin/python demo/demo_app.py`）；**agent 杀不掉非本会话进程**（分类器拦），要让用户重启。做 §3 #1（自动解析）前先 `curl -sX POST http://127.0.0.1:5000/api/parse_case -H "Content-Type: application/json" -d '{"text":"x"}'` 确认返 JSON。
- **AirPlay 占 `*:5000`** → app API_BASE 用 `127.0.0.1:5000`；后端起要 `HOST=127.0.0.1`。
- **openFDA 默认开**（`use_openfda` 硬编码 true）→ analyze 实时打 api.fda.gov、要联网；真标签结构化剂量未验证 → 药卡 dose 显示 "withheld"（正常，非 bug）。
- **模拟器往 TextInput 打字触发 iOS 重音弹窗** → 用 `echo '文本' | xcrun simctl pbcopy <udid>` + `cmd+v` 粘贴。
- 偶发 `auth.neutron.health`（Photon OAuth）应用内浏览器弹窗 = 遗留 auth 会话，重启 app 消失，与 UI 改动无关。

## 6. 提交 / PR

- 续做也走 `feat/voice-input`；打磨完照样拆干净 commit。
- **未 push**。真要给 Chun review：`case_demo.html`(a5ed741)、`reasoning.tsx`/`prescribe.tsx`/后端 是他的范围 → cherry-pick 相关 commit 到单独分支开 PR（base=`feature/mobile-demo`，本机无 `gh`，用 compare URL）。
