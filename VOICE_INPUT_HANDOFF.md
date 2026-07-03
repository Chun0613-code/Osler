# VOICE_INPUT_HANDOFF —— 语音病历录入

> 更新于 2026-07-02(SESSION 3:3A 转写 + 3B 结构化确认均完成并验证)。**架构从"设备端实时识别"改成了
> "录音+云端转写"(SESSION 2,原因见 §2)。给新 session 接手用:先读这份 + 记忆里的 [[oslian-osler-project]]、
> [[oslian-eprescribing]]。文中 file:line / 函数名动手前对现有代码验一遍。

---

## 0. TL;DR —— 现在到哪了

**阶段 3A 已完成并在模拟器端到端验证通过**:医生点话筒 → 录音(expo-audio)→ 上传后端 → Gemini 云转写
→ 文字进 CASE NOTE。含药名(实测 "Lisinopril" 都对)。后端 `/api/voice/transcribe` 实测 3 次 200。

**阶段 3B 已完成并在模拟器端到端验证通过(2026-07-02 SESSION 3)**:转写 settle → `/api/parse_case` →
**CAPTURED chips**(indication/age/sex/eGFR/allergies/meds,amber 虚线=待确认,tap→写进 Case-details 表单+✓)
+ **VITALS · AUTO-FLAGGED BY THE ENGINE**(⚠️ Hypotension SBP 88<90、⚠️ Tachycardia HR 112>100,来自
`patient_profile.flags()`)+ "nothing is prescribed from voice alone"。兑现样机。Analyze 时 vitals 经 CASE NOTE
文本被 `agent.analyze` **重解析**送进引擎,Reasoning 页确实带 hypotension/tachycardia 旗标(预览==引擎实际所见)。
详见 **§3B**。

**阶段 4 进行中**:3A + 3B 已 commit(`cd49217`)+ push 到 `origin/feat/voice-input`。**只差在浏览器点建 PR**
(base=`feature/mobile-demo`,compare URL 见 §8;`/api/parse_case` 是 Chun 的后端范围,PR 里请他 review)。

**3B 微调(SESSION 3 续,已提交)**:缺字段提示 / indication 建议 chip / 未确认轻提醒 / **五字段必填门槛**(过敏·用药可空)
/ 去装饰 emoji —— 详见 **§3B › 3B 微调 + 微调②**。均已在模拟器验证并 commit(在 `cd49217` 之后)。

**⏭️ 下个 session = 前端设计重构**:用户觉得 Analyze 屏"功能全挤一块儿了太杂"(sample cases + 表单 + 语音 + CAPTURED
卡 + openFDA + Analyze 全堆一页)。下次要重排信息层级 / 分屏 / 收纳。**功能已齐且验证过,重构别把上面这些逻辑改坏**。

- 分支 **`feat/voice-input`**(从 feature/mobile-demo 切);已推送。PR(base=`feature/mobile-demo`)仍待在浏览器点建(见 §8)。
- ⚙️ **本机当前运行态**(agent 起的,可能已停):app 指向 **:5000**(用户真后端,Photon 已连,但**旧代码→语音 parse_case 404**);
  Metro :8081(watch);另有 agent 的测试后端 :5055(新代码)。要语音也通:**重启 :5000 后端加载新代码**(会清 Photon 登录、需重连一次)。
- 怎么跑:见 **§4**。踩过的坑:见 **§6**(尤其 5000 端口 AirPlay + 模拟器打字触发重音弹窗 → 用剪贴板粘贴)。
- ⚠️ **要用这功能,用户的 5000 后端必须重启**加载新 `/api/parse_case` 路由(agent 起的是 5055 测试实例)。

---

## 1. 这功能是什么
医生说几句话 → 转成文字 → 填进病历 → 出药物推荐。理解病例的"大脑"早就有(`demo/case_parser.py`
`parse_llm`、`engine/patient_profile.py` `missing_core()/flags()`),语音只是加个前端入口。

---

## 2. ⚠️ 架构大转向(SESSION 2 的核心发现)

**初版计划(设备端 `expo-speech-recognition` / iOS SFSpeechRecognizer 实时流式)在 iOS 模拟器上根本跑不了。**
实测:任何模式(实时麦克风 / 文件转写、on-device / server)都报同一个错:

```
error: audio-capture: Failed to initialize recognizer
```

根因:**iOS 模拟器不支持 SFSpeechRecognizer 的音频子系统**(macOS 隐私设置里根本没有 "Simulator"
麦克风项 = 它从没成功请求过)。库文档也写了:文件转写不支持时同样抛 `audio-capture`。

**对比 Glossa 为什么能在模拟器跑语音**:Glossa 用 `expo-audio` **录音成文件 → 上传后端 → 云端转写**
(`AVAudioRecorder` 录音在模拟器是好的;坏的只是 `SFSpeechRecognizer` 实时识别)。→ **Osler 改走同款路线**。

**已定架构(用户拍板)**:
- **录音**用 `expo-audio`(模拟器可用) → **上传** → **后端 Gemini 云转写**。
- **v1 = 录完再出字**(不是逐字流式),但录音时保留 Listening 卡 + 波形 + 计时的视觉。真正逐字流式要
  上 **Deepgram streaming**(留作后续,`demo/transcribe.py` 已抽象好 `VOICE_STT_PROVIDER`,小改即可)。
- **隐私**:云转写 = 患者音频上云。Osler 只做 sandbox demo(无真实患者)→ demo 阶段 OK;生产要 HIPAA
  BAA 或真机走设备端。代码里已注释标注。
- **`expo-speech-recognition@sdk-54`(3.1.3)仍装着但当前未使用**——它只有**真机**能跑(设备端、保隐私),
  留作以后真机上的可选路径。要清理的话删它 + app.json 的 plugin,重建即可。

---

## 3. 已完成(阶段 3A)—— 改了哪些文件(都在 feat/voice-input,未 commit)

**后端**
| 文件 | 改动 |
|---|---|
| `demo/transcribe.py`(新) | Gemini 音频转写。`transcribe_audio(bytes, fmt)` + `transcribe_available()`。provider 抽象(`VOICE_STT_PROVIDER` 默认 gemini,`_deepgram_transcribe` 是留的坑)。复用 `google-genai` SDK(.venv 已装)+ `GEMINI_API_KEY`。 |
| `demo/demo_app.py` | 加 `@app.route("/api/voice/transcribe", POST)`(收 multipart `audio` + `format` → 调 transcribe.py → `{"text":...}`);`MAX_CONTENT_LENGTH` 从 1MB 提到 **8MB**(容纳音频)。 |
| `demo/.env` | 加了 `GEMINI_API_KEY=`(用户的通用 Google API key,实测能转写)。 |

**前端**
| 文件 | 改动 |
|---|---|
| `mobile/src/hooks/useVoiceCapture.ts`(新) | expo-audio 录音 → 上传 hook。tap-to-start/stop、60s 上限、错误态。**照抄 Glossa 的 `useVoiceInput.ts`**(含卸载时守护 recorder 的坑)。 |
| `mobile/src/components/VoiceDictation.tsx`(新) | Listening 卡 UI(navy 卡 + 橙话筒 + 波形 + 计时 + Stop + 安全提示)。idle 态是橙色 "🎙 Dictate case note" 按钮。 |
| `mobile/src/api/osler.ts` | 加 `transcribeVoice(uri, fmt)`(FormData multipart 上传到 `/api/voice/transcribe`)。 |
| `mobile/src/app/analyze.tsx` | CASE NOTE 上方渲染 `<VoiceDictation onTranscript=... />`,转写追加进 `text`(CASE NOTE)。 |
| `mobile/package.json` | +`expo-dev-client`(~6.0.21)、+`expo-audio`(~1.1.1)、+`expo-speech-recognition`@sdk-54(3.1.3,未用);**删了 `@expo/ui`**(见 §6)。 |
| `mobile/app.json` | +`expo-speech-recognition` & `expo-audio` config plugin + 麦克风/语音权限文案;prebuild 自动写入 `bundleIdentifier: com.alphonse1223.oslian-rx` + Android RECORD_AUDIO。 |

**验证**:`npx tsc --noEmit` 通过;dev build 编过、装进 iPhone 17 模拟器;端到端录音→转写→进病历实测 OK
(3× `POST /api/voice/transcribe` 200)。

---

## 3B. 已完成(阶段 3B)—— 转写→结构化→确认(都在 feat/voice-input,未 commit)

**后端**
| 文件 | 改动 |
|---|---|
| `demo/demo_app.py` | 加 `import case_parser`;加 `@app.route("/api/parse_case", POST)` + helper `_indication_label`。收 `{text, api_key?, provider?}` → `case_parser.parse` → `agent.build_patient().flags()/missing_core()` → 返回 `{fields, parser, indication, indication_label, indication_known, flags, missing_core}`。空 text→400。**不做 analyze/prescribe**。 |

**前端**
| 文件 | 改动 |
|---|---|
| `mobile/src/api/osler.ts` | 加 `parseCase(text, llm?)` + 类型 `CaseFlag`/`ParsedCaseFields`/`ParseCaseResult`。**与 `analyze()` 传同样的 `api_key`/`provider`**(见下"一致性")。 |
| `mobile/src/components/CapturedFields.tsx`(新) | CAPTURED 卡:值 chips(amber 虚线=pending,tap→confirm 变 navy+✓)+ "Confirm all" + parser 徽标(RULES/AI)+ VITALS amber 旗标行(⚠️ + 人话 flag 名 + detail)+ indication 不可映射/空 提示 + 底部安全语。导出 `capturedKeys(parsed)` 让 analyze.tsx "Confirm all" 与 chip 集合同源(不分叉)。 |
| `mobile/src/app/analyze.tsx` | 加 state `parsed/parsing/confirmed` + `textRef`;`runParse()`;`handleTranscript`(转写→追加 CASE NOTE→自动 `runParse`);`confirmField(key)`(写进 `manualFields`,sex 归一化 M/F/Other)+ `confirmAllFields`。JSX:`VoiceDictation onTranscript={handleTranscript}`;textarea 下加 **"⚙ Structure note"** 按钮(给打字/重解析,也是不靠语音的测试入口)+ `<CapturedFields>`。 |

**关键设计决策(别再纠结)**
1. **一致性铁律**:`/api/parse_case` 与 `/api/analyze` 用**同一个 provider**(pass-through `api_key`/`provider`)。因为 `agent.analyze_stream` 在 Analyze 时会**重解析 CASE NOTE 文本**并合并(表单字段覆盖),所以 CAPTURED 预览必须与那次解析一致,否则"预览显示的 flags"≠"引擎实际所见"。→ 预览用啥 provider,analyze 就用啥。
2. **vitals 怎么进引擎**:表单没有 vitals 输入框。BP/HR/SpO2 **不做 chip**,只当 VITALS 旗标显示;它们经 CASE NOTE 文本被 analyze 重解析送进引擎(实测 Reasoning 页带 hypotension/tachycardia)。所以**别删 CASE NOTE 里的转写**。
3. **demo 默认走 rules 解析**:`.env` 只有 `GEMINI_API_KEY`、无 `OPENAI_API_KEY`/`LLM_PROVIDER` → `llm_client.available()` 默认 provider=openai→无 key→**rules**。rules 对标准 demo case 完美(age/sex/BP/HR/eGFR/indication 全中)、确定性、零延迟。**要 LLM 解析**(更适合乱说的口语):`.env` 加 `LLM_PROVIDER=gemini` 即可,parse_case + analyze 一起切,保持一致。
4. **rules 解析对 run-on 句子会犯迷糊**:实测 "allergic to penicillin, taking aspirin"(无句号)→ allergies 抓成 "penicillin, taking aspirin"。**这正是 tap-to-confirm 存在的理由**——医生看到就改。不是 bug,是安全设计的活教材。LLM 解析会切干净。
5. **tap-to-confirm ≠ 硬闸**:confirm 把值写进 Case-details 表单(可见+可编辑);未 confirm 的值仍会经文本重解析到达引擎。安全属性靠"全部可见"(转写+chips+flags)+ "语音永不触发 Sign & Send" 保证,不是靠阻断。sandbox demo 可接受。

**验证(2026-07-02 SESSION 3)**:`npx tsc --noEmit` exit 0;Metro bundle 1270 modules 无错;curl `/api/parse_case` 3 case 全对
(ACS→hypotension+tachycardia、症状→asthma 映射、空→400);模拟器端到端 **打字→Structure→CAPTURED(6 chips+2 amber
旗标)→Confirm all(全变 ✓ 且表单填好 indication/age/sex/eGFR/allergies/meds)→Analyze→Reasoning 带 hypotension+
tachycardia**。截图在 scratchpad(`verify_captured_card.png`、`verify_reasoning_flags.png`)。

### 3B 微调(同 SESSION 3,**在 `cd49217` 之上,尚未 commit**)—— 三个"后端已有数据/接口、补前端"的小项
1. **缺字段提示**:CAPTURED 卡渲染 `missing_core`("⊘ Not captured: weight · eGFR — optional…")。数据早在
   `/api/parse_case` 返回,只是之前没显示。`CapturedFields.tsx`。
2. **indication 建议 chip**:`indication_known=false` 时,除了 amber 提示,再给一排**可点的已知病种**(横向滚动,
   `getCases` 返回的 `indications` 之前被 analyze.tsx 忽略,现在存下来传进去);点一个 → 填进表单 indication +
   标 `confirmed('indication')` → 提示塌成 "✓ Indication set"。**消灭唯一会 400 的硬卡点**。另:`capturedKeys` 改成
   **只在 `indication_known` 时才给顶部 indication chip**(否则确认一个映射不了的诊断只会让 Analyze 400)。
   `analyze.tsx`(state `indications` + `pickIndication`)、`CapturedFields.tsx`(props `indicationOptions`/`onPickIndication`)。
3. **未确认轻提醒**:带 pending chip 点 Analyze → `Alert.alert("Unconfirmed values","N … not confirmed yet")`
   [Analyze anyway]/[Review]。**不阻断**(兑现安全铁律 §7.1"可见确认")。`analyze.tsx`:`onAnalyze` 拆成 guard
   包 `runAnalyze`;imported/preset(overrideCase)跳过提醒。

**验证**:`tsc` exit 0;fresh Metro bundle 1282 modules 无错;模拟器实测三条全过(ACS→缺 weight 提示 + 4 项未确认
Alert;58F 无诊断→picker→点 ACS→表单 indication 填成深色实值 + "✓ Indication set")。截图 `verify_batch_indication_set.png`。

### 3B 微调② —— 必填门槛 + 去 emoji(SESSION 3 续,同一批未 commit → 本次一起提交)
4. **必填硬门槛**(`analyze.tsx`):`onAnalyze` 加校验 —— **Indication / Age / Sex / Weight / eGFR 五个必填**才能
   Analyze(`REQUIRED_FIELDS` + `missingRequiredFields`;缺 → 红字列出、不分析、不弹覆盖层)。**团队策略(和 Chun 定,
   2026-07-02)**:精确剂量要 weight/eGFR,未成年区分要 age,性别区分要 sex。⚠️**过敏 / 当前用药不必填**(留空=无,
   Alphonse 明确纠正过——别再把这俩设成必填)。提取 `collectAnalyzeInput` 给门槛与请求共用字段合并。CASE DETAILS 下加必填提示行。
5. **去装饰 emoji**(用户嫌"太 AI",见记忆 [[avoid-decorative-emoji-in-ui]]):按钮 `Structure note`(去 ⚙)、
   `Dictate case note`(去 🎙,`VoiceDictation.tsx`);AGENT WORKFLOW 覆盖层去 ⚡ + 每步图标统一渲染中性 `•`
   (`analyze.tsx` 里直接渲染 `•`、忽略 `s.icon`,连 Chun 后端 trace 发来的 emoji 也一并盖掉,不用改后端)。**保留 ⚠️**
   (VITALS 临床告警,功能性)。`✓`/`⊘`/`→` 暂留(待用户定夺,见对话)。

**验证②**:`tsc` exit 0;模拟器实测 —— 26F asthma(**meds 空**)→ Analyze 放行(不再拦 meds);缺字段案例点 Analyze →
红字 "missing: …";覆盖层 `•` 圆点无 emoji。

---

## 4. 怎么跑 / 验证

**要 dev build,不能用 Expo Go**(expo-audio/expo-speech-recognition 是原生模块)。CocoaPods 已用
Homebrew 装好(`pod` 1.16.2;系统 ruby 2.6 太老,brew 的自带新 ruby)。

```bash
# 后端(必须 HOST=127.0.0.1,见 §6 的 AirPlay 坑)
cd ~/Desktop/codeworks/SouforgeTechCodeworks/Osler
HOST=127.0.0.1 PORT=5000 .venv/bin/python demo/demo_app.py   # banner 应打印 Running on http://127.0.0.1:5000

# App:改了原生模块要重建;只改 JS 用 start --dev-client 热更
cd mobile
eval "$(/opt/homebrew/bin/brew shellenv)"; export LANG=en_US.UTF-8
npx expo run:ios --device AF02103D-2418-45DE-85C5-AED1472E0A54   # 首次/加库重建
# 之后日常:npx expo start --dev-client   （连已装的 dev build,JS 热更）
```

- **iPhone 17 / iOS 26.5 模拟器 UDID**:`AF02103D-2418-45DE-85C5-AED1472E0A54`
- **深链**:`xcrun simctl openurl <udid> "oslianrx://analyze"`(截图前留 1~2 拍,导航有延迟);根路由
  `oslianrx://` 会弹到 dev-client launcher,别用。
- **自测后端转写**(不用手机):`say -o s.aiff "..." && afconvert -f m4af -d aac s.aiff s.m4a` 造个 clip,
  再 `curl -X POST -F "audio=@s.m4a" -F "format=m4a" http://127.0.0.1:5000/api/voice/transcribe`。
- **模拟器录音**:Simulator 菜单 **I/O → Audio Input** 选 Mac 麦克风,然后对着麦克风说(本机实测能录到音)。

---

## 5. ✅ 阶段 3B 已完成(实现细节见 §3B)—— 样机对照 & 遗留说明

**目标样机(已兑现)**:
- **CAPTURED** 区:结构化 chip(indication / age / sex / eGFR / allergies / meds)+ "tap a value to confirm"。✅
- **VITALS → AUTO-FLAGGED BY THE ENGINE**:⚠️ Hypotension SBP 88<90、⚠️ Tachycardia HR 112>100
  —— `engine/patient_profile.py` 的 `flags()`(amber 色)。✅
- 底部 "nothing is prescribed from voice alone"。✅ 「Review and analyze →」= 沿用现有 "Analyze case →" 按钮。

**没做(有意):** TRANSCRIPT 里逐字高亮数字 chip(样机的第一块)。当前用 CAPTURED 卡承载"看到什么被抓到了",
CASE NOTE textarea 显示原文可编辑。要做逐字高亮得 tokenize 转写并渲染富文本,成本高、收益低,留作以后。

**注意(仍成立,非 bug)**:Analyze 引擎**必须**有能映射的 indication(固定已知表:ACS、acute heart failure、
hypertension、asthma…)。CAPTURED 卡在 `indication_known=false` 时会给 amber 提示 **+ 一排可点的已知病种 chip**(点一个直接填进表单 indication,
见 §3B 微调 #2)。纯症状("shortness of breath")→ 推不出 → 点个建议病种即可 Analyze。(实测 "wheezing" 被 rules 命中 asthma。)

**阶段 4 状态**:`tsc --noEmit` 已过;已 commit `cd49217`(3A+3B 一个 commit)+ push `origin/feat/voice-input`。
**剩:浏览器点建 PR**(base=`feature/mobile-demo`,URL 见 §8;`/api/parse_case` 属 Chun 后端,PR 里让他 review)。

---

## 6. 本 session 踩的坑 / 关键事实(别重查)

1. **`@jamsch/expo-speech-recognition` 已废弃 → 改名 `expo-speech-recognition`**,按 SDK 打 tag。SDK 54 用
   `@sdk-54`(=3.1.3)。**别装 `latest`(=56.x,给 SDK 56 的,会炸)**。
2. **删了 `@expo/ui`**:一个**没被任何源码 import** 的 canary 依赖,是首次原生编译**唯一**的报错源(引用了
   `ExpoModulesCore@3.0.30` 里不存在的 `ExpoSwiftUI.RNHostViewProtocol`)。删掉零影响。**团队若要用 @expo/ui,
   得换 SDK-54 兼容版**——让 Chun 知道。
3. **5000 端口 = AirPlay 地雷**:macOS AirPlay Receiver(ControlCenter)占 `*:5000`。Flask 靠 SO_REUSEADDR
   仍能绑 `127.0.0.1:5000`(所以启动一定给 `HOST=127.0.0.1`)。**app 的 API_BASE 用 `127.0.0.1:5000` 不能用
   localhost**(iOS 把 localhost 解析成 ::1 → 命中 AirPlay)。curl 测也用 127.0.0.1。
4. **残留后端**:上个 session 留了个 `/tmp/osler_cors_runner.py`(给 web 加 CORS 的临时包装)占着 5000、且是
   旧代码(没新路由)。要 `pkill -f osler_cors_runner` 清掉(**agent 被分类器拦,不能杀非本 session 起的进程
   → 让用户跑**)。CORS 只有跑 expo --web 才需要;模拟器原生无 CORS,直连即可。
5. **模拟器 SFSpeechRecognizer 全废**(§2);但 **expo-audio 录音在模拟器好用**。
6. **Gemini key**:Glossa 那把在 Joshua 的部署后台,**本地找不到**(仓库只有 `.env.example` 空占位)。用户给了自己
   的通用 Google API key(`AQ.Ab8…` 格式,不是常见的 `AIza…`,但**实测能转写**)。已写进 `demo/.env`。
7. Gemini 转写偶发把 "eGFR" 听成 "EG for"(合成语音更明显,真人清楚些)——正是 3B 高亮待确认要兜的。
8. **模拟器里往 RN TextInput 打字会触发 iOS 长按重音弹窗**(À Á Â… 只进一个字符),即使 I/O→Keyboard→
   Connect Hardware Keyboard 已勾。原因是逐键按住时长被 iOS 当成"选重音"。**绕法:`echo '文本' | xcrun simctl
   pbcopy <udid>` 设剪贴板 → 点输入框 → `cmd+a` `cmd+v` 粘贴**(硬件键盘已连,cmd 快捷键直达 iOS)。SESSION 3
   靠这个把测试病历塞进 CASE NOTE。
9. **测 3B 不用语音**:点 textarea 下的 **"⚙ Structure note"** 按钮就能对当前文本跑 `/api/parse_case`(语音
   转写 settle 也会自动跑同一个)。省得在模拟器折腾麦克风。

---

## 7. 安全铁律(不可破)+ PHI
1. 数字 / 药名 / 过敏必须可见确认,绝不静默采纳(3B 高亮)。
2. 语音永远不能直接触发 Sign & Send。
3. 转写原文存下来当溯源。
4. **PHI**:云转写把患者音频送 Google——sandbox demo 可接受;生产需 BAA 或真机设备端。

## 8. 协作 / PR(见 [[oslian-osler-project]])
- 从 `feature/mobile-demo` 切 `feat/voice-input`,push 后开 PR:**base = `feature/mobile-demo`(不是 main)**。
- 本机没 `gh` → 网页 compare URL:
  `https://github.com/Chun0613-code/Osler/compare/feature/mobile-demo...feat/voice-input?expand=1`

## 参考
- 样机 = 用户在另一 session 展示的 Analyze 语音图(navy Listening 卡 + 高亮数字 + CAPTURED chips + amber 生命体征旗标)。
- 代码:`demo/transcribe.py`、`demo/case_parser.py`、`demo/demo_app.py`、`engine/patient_profile.py`(flags/missing_core)、
  `mobile/src/app/analyze.tsx`、`mobile/src/hooks/useVoiceCapture.ts`、`mobile/src/components/VoiceDictation.tsx`、`mobile/src/api/osler.ts`。
- Glossa 参考实现:`glossa/mobile/hooks/useVoiceInput.ts`、`glossa/backend/voice_service.py`。
- 电子处方续做另见 `Osler/PRESCRIBING_HANDOFF.md`。
