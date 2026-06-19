# Osler·Rx 电子处方整合 — HANDOFF

> 给下一个 session 接力用。读这份 + 代码就能继续，无需上文。
> 仓库：`SouforgeTechCodeworks/Osler`，分支 **`feature/mobile-demo`**（不是 SoulLink 目录）。
> 详细背景计划：`~/.claude/plans/hi-claude-code-alphonse-oslian-app-anita-cozy-leaf.md`

## 一句话状态
✅ **2026-06-18 端到端跑通并验证**："症状→符号引擎荐药→**原生一屏 Sign & Send**→`createPrescription`+`createOrder` 送药房" 全闭环在 Photon 沙箱实跑成功。用 Alphonse 的授权身份(`can_prescribe=True`)真建了记录:处方 `rx_01KVF3WJ8CW25M7G0SX6850H63`、订单 `ord_01KVF3WMQQPE88SJR3S40PD54G`、患者 `pat_01KVF3V8119BYSR3839SY05232`(可在 app.neutron.health 后台查到)。

> 历程:原生开方代码写完 → 修掉 SPA client id 的 `l/I` 错字(关键发现 #7) → app 内 OAuth 登录拿到含 `write:prescription` 的 provider token → Sign 第一次报 `refillsAllowed 已废弃`,改用 `fillsAllowed=refills+1`(关键发现 #8) → 再 Sign 成功。**功能完成。剩下都是打磨,不是必须。**

---

## ✅ 已完成并验证
**后端 `demo/`**（Flask :5000，启动读 `demo/.env`，打印 `photon=on (sandbox)`）
- `photon_client.py` — Photon M2M OAuth(stdlib urllib)、`graphql()`、`create_or_sync_patient()`（**实跑通过**，返回真实 `pat_…`）、`search_medications()`（**实跑通过**）、`create_prescription()`/`create_order()`（已写好，**但 M2M 无权限调用——见下**）、`prescribe_url()`(托管页 fallback)。
- `prescription.py` — 候选药→处方记录、安全门控（`avoid/block` 禁开）、管制药排除清单、内存存储 `_RX`、mock 模式。
- `prescribe_embed.html` — WebView 用的 mock 表单（无密钥时）；Photon Elements 分支已废弃（见下）。
- `demo_app.py` — `/api/prescribe/start|context|complete`、`/prescribe-embed`、`/api/photon/webhook`、`/api/prescriptions`；`.env` 自动加载。

**手机端 `mobile/src/`**（Expo/RN，6 tab）
- `components/DrugCard.tsx` — Prescribe CTA（`avoid/block` 隐藏）。
- `app/prescribe.tsx` — 当前是"系统浏览器开 Photon 托管页"（**下一步要改成原生**）。
- `app/prescriptions.tsx` — 新 Rx tab；`components/PrescriptionCard.tsx`。
- `api/osler.ts`、`app/_layout.tsx`（注册 Rx tab + 隐藏 prescribe 路由）、`app/reasoning.tsx`（接线）。
- tsc 干净；模拟器实测：患者同步到 Photon 沙箱 OK、系统浏览器能加载到 Photon 登录页。

### ✨ 2026-06-18 新增：原生 provider-OAuth + Sign & Send（已写完，未做交互测试）
架构定调：**整个 OAuth 在后端跑，provider token 只存后端内存，设备只知道 connected:true/false**。
理由：① 回调 URL 固定 `http://127.0.0.1:5000/api/photon/oauth/callback`，Expo Go / dev build / 换机器都一样，比每台机器变的 `exp://` 好加白名单；② 所有密钥/令牌不落地设备（符合"密钥逻辑集中"）；③ 手机端**零新依赖**（复用已装的 expo-web-browser + expo-linking）。
- **后端 `demo/photon_client.py`**：`graphql()`/`create_prescription()`/`create_order()` 加可选 `token=` 参数 → 患者同步/搜药仍用 M2M，开方/下单用 provider token。
- **后端 `demo/photon_oauth.py`（新）**：纯 stdlib 的 PKCE（secrets+hashlib+base64）、`_PENDING` state 存储、`_PROVIDER` token 存储+refresh、`build_authorize_url()`、`exchange_code()`、`provider_token()`、`status()`、`disconnect()`。Auth0 端点由 OIDC discovery 实证：authorize=`https://auth.neutron.health/authorize`、token=`.../oauth/token`、S256 PKCE 支持。scope 请求含 `write:prescription`（Auth0 按 provider 角色发实际子集）。带 `organization=org_…` 预选机构。
- **后端 `demo/prescription.py`**：`options(rx_id)`（搜 20 条→**偏好纯仿制药启发式** `_rank_candidates`→返回 top10 + 默认 sig/qty/unit/days/refills；`_dispense_unit_for_form` 按剂型给默认单位）、`sign(rx_id, provider_token, …)`（再过一遍安全门控 → createPrescription→createOrder→complete）。
- **后端 `demo/demo_app.py`**：新路由 `/api/photon/oauth/{start,callback,status,disconnect}`、`/api/prescribe/{options,sign}`。callback 用 302 把 `oslianrx://photon-connected?photon=connected|error&msg=` 深链回 app（已 curl 实测 302+Location 正确）。
- **手机端 `mobile/src/api/osler.ts`**：`getPhotonStatus/getPhotonAuthorizeUrl/disconnectPhoton/getPrescribeOptions/signPrescription` + 类型。
- **手机端 `mobile/src/app/settings.tsx`**：新 "PHOTON · E-PRESCRIBING" 卡 → Connect（`Linking.createURL('photon-connected')` 作回调 → `WebBrowser.openAuthSessionAsync`）/ Disconnect / 显示已连医生名+邮箱；focus 时刷新状态。
- **手机端 `mobile/src/app/prescribe.tsx`**：三态分支——mock→旧 WebView 原样保留；photon+未连→提示去 Settings 连接；photon+已连→**原生审核屏**（药品单选列表、sig 多行、数量/单位/天数/refills 可改、Sign & Send → `/api/prescribe/sign` → 回 Rx tab）。旧"跳系统浏览器开方"分支已删。
- **验证到的程度**：tsc=0；后端 import/启动 OK，banner 打印 `photon=on (sandbox) provider_oauth=ready`；M2M token+搜药实跑通；authorize URL 形态正确且能打到 Auth0；OAuth callback 深链 302 实测正确。**没做的**：真正 provider 登录 + 真开一张沙箱处方（要你的 Neutron 账号交互登录，见下方"怎么测"）。

---

## 🔑 关键发现 / 架构定调（别再重新踩）
1. **M2M 令牌不能建处方**。当前 M2M scope = `read:patient write:patient read:prescription read:order write:order`，**没有 `write:prescription`**（Photon 故意如此——开处方必须绑认证过的医生身份，不让后端机器代开）。`createPrescription` 实跑报 `MISSING_PERMISSIONS: write:prescription`。
   → **结论架构：医生一次性 OAuth 登录（系统浏览器）→ app 拿到 provider 用户令牌 → 用该令牌原生调 `createPrescription`+`createOrder`。浏览器只用于偶尔登录，开方全程原生、不跳。**
2. **嵌入式 WebView 登录此路不通**：Auth0/Google 封禁嵌入 WebView（试过，报 Neutron "something went wrong"）。**系统浏览器**登录正常（已验证 Photon 登录页能在系统浏览器加载）。
3. **没有真实持照/带 DEA 号的开方医生**（团队现状）→ **现在任何真实开药都做不了（不止管制药）**，只能 sandbox 演示。真上线（任何药）卡在"招到一个 prescriber"这个商务步骤。
4. **管制药/EPCS：搁置**。需真实医生 + Experian 身份核验(IAL2) + 2FA 令牌 + DEA 第三方软件审计（或把签名这步路由到 Photon 已认证流程=必然"跳"）。架构上：检测 `Medication.controlled` → 走 Photon 认证 2FA 流程。**现在不做**。
5. 杂项坑：① 搜药别请求 `controlled` 字段（沙箱有 null 脏数据破坏 non-null 查询）；`Medication.id` 即 `treatmentId`。② `createOrder` 省略 `pharmacyId` → Photon 自动路由；`address` 用于路由。③ patient `phone` 必须合法 NANP 区号（`555` 被拒，用 `+12025550102`）。④ 内存缓存：analyze 与 prescribe 须在同一次后端运行内；app 整页 reload 清空患者→需重新分析。⑤ AirPlay 占 :5000 → 用 `127.0.0.1` 非 localhost（已配）。
6. **写操作会被 auto-mode 安全分类器拦**：直接用 Bash 跑 `createPrescription`/`createOrder` 会被拦（写外部系统）。要么经 app 后端在 UI 操作时触发，要么先取得用户明确授权。
7. ✅ **（2026-06-18 已解决）SPA client id 是 `l`→`I` 错字**。`.env` 原值 `0ltrnfbl…HAEwl`（第 8、32 位是小写 l），dashboard 复制按钮的真值是 `0ltrnfbI0r9ku2hwR89TQkmOrSyHAEwI`（第 8、32 位是**大写 I**）。改对后 authorize URL 实测 HTTP 200 直接进 `app.neutron.health/login`。诊断路径留档：拿 `.env` 值打 `auth.neutron.health/authorize` 报 `Unknown client`，且在 neutron/photon/boson 三个 Auth0 租户都报同样错；又从 Photon **官方 SDK**（`@photonhealth/sdk@1.3.4`，`cdn.jsdelivr.net/npm/@photonhealth/sdk/dist/lib.js`）确认登录就是用 `auth.neutron.health` + 调用方 clientId + audience `api.neutron.health` 的标准 Auth0 PKCE——所以方法没错，只是 client id 值坏。让用户用复制按钮粘出真值，diff 出 `l/I` 之差。
   → **连带纠正上一份的发现 #2**："嵌入式 WebView 登录报 Neutron 'something went wrong'" **十有八九就是这个 Unknown-client 错误页**（Auth0 的 "Oops!" 页就长这样），不是 Auth0 封 WebView。
   → **回调用裸 origin**：Photon SPA 的 "Whitelisted URLs" 是裸 origin（无 path/query），`http://127.0.0.1:5000` 已在白名单里。所以 `REDIRECT_URI` 改成裸 `http://127.0.0.1:5000`，由 Flask **根路由 `/`** 检测 `?code&state` 接管 OAuth（`demo_app.index`），**后台零改动**。真机换局域网 IP 时，把 `http://<LAN_IP>:5000` 也加进 Whitelisted URLs，并设 `PHOTON_OAUTH_REDIRECT` 环境变量。
8. ✅ **（2026-06-18 实证）`createPrescription` 不能再传 `refillsAllowed`**。Photon 报 `INVALID_PRESCRIPTION: refillsAllowed has been deprecated. Please create a prescription using only fillsAllowed.` 已把 `photon_client._CREATE_RX` 改成只传 `fillsAllowed=refills+1`（总配药次数=首次+续配）。改完 Sign 成功。`createOrder`(省略 pharmacyId → Photon 自动路由)实测**一次过**。
9. ⚠️ **provider token 存后端内存(`photon_oauth._PROVIDER`)、进程级**：重启后端就丢，要在 app 里重点一次 Connect Photon（Auth0 记会话→秒连、不用再输密码）。若嫌烦可加落盘持久化(`.photon_session.json` gitignore)——属打磨项，没做。

---

## 🚧 下一个 session 主任务：~~非管制药"原生一屏开方"~~ ✅ 已做完 → 现在是"配置 + 测一次"

**原生开方代码已全部写完 + SPA client id 已修对 + 回调已改裸 origin**（见上方 ✨ 新增块、关键发现 #7）。下一个 session 不用再写功能、不用改后台，只需：
1. 跑一次端到端交互测试（见"怎么测"）——这是现在唯一剩的事。
2. 若 createPrescription 报缺字段/`dispenseUnit` 非法/某药不可开，按 502 里 Photon 返回的 GraphQL 错误信息微调 `photon_client._CREATE_RX` 参数或换个药即可（错误信息原样透传到 app 弹窗 + 后端日志）。
3. 前提：登录用的 Neutron 账号得是**有开方权限的 prescriber**，否则拿不到 `write:prescription`、sign 会 502。

<details><summary>（存档）原始主任务清单 — 现已实现，留作对照</summary>

## 🚧 下一个 session 主任务：非管制药"原生一屏开方"
目标：去掉"跳浏览器开药"，改成 **医生登录一次 → 之后原生 Sign & Send**。

**① 后端（`demo/`）**
- Provider OAuth（拿带 `write:prescription` 的用户令牌）：实现 Photon SPA 的 OAuth2 **PKCE** 授权码流程。授权端点 `https://auth.neutron.health/authorize`，audience `https://api.neutron.health`，client_id = `PHOTON_SPA_CLIENT_ID`（在 `.env`），scope 需含 `write:prescription`。app 用系统浏览器完成、拿 code 换 token，token 存设备（`expo-secure-store`）。
- `photon_client.graphql()` 现在只用 M2M token；要支持"传入 provider token"调用（createPrescription/createOrder 用 provider token，患者同步/搜药仍可用 M2M）。
- 新端点：`GET /api/prescribe/options?drug=`（返回 `search_medications` 候选 + 默认 sig/qty/days/refills）；`POST /api/prescribe/sign`（用 provider token：createPrescription→createOrder→更新记录）。
  - 注意：provider token 在 app 端，所以"用谁的 token 调"要定：app 直接调 Photon GraphQL，或 app 把 token 传后端由后端调。二选一（建议后端调，密钥逻辑集中）。

**② 手机端（`mobile/src/`）**
- `app/settings.tsx`：加 **"Connect Photon"** → `expo-web-browser` 的 `openAuthSessionAsync`（自定义 redirect scheme）跑 OAuth → 存 provider token。
- `app/prescribe.tsx`：把 WebBrowser/WebView 那套换成**原生审核屏**：药品/规格选择（来自 options）、可改 sig/数量/天数/refills、药房（可留空=自动路由）、**Sign & Send** → `POST /api/prescribe/sign` → 回 Rx tab。

**③ 验证**：Settings 连一次 Photon（沙箱测试账号即你的 Neutron 登录）→ 分析 GERD → 原生开 omeprazole → Neutron 后台 + app Rx tab 都看到订单。
（注：实跑 createPrescription 会在沙箱建真实测试记录、不送真药房；需用户授权或经 app UI 触发。）

</details>

---

## ❓待解决的技术问题
- ✅ **OAuth 参数已实证**（OIDC discovery `https://auth.neutron.health/.well-known/openid-configuration`）：authorize=`/authorize`、token=`/oauth/token`、S256 PKCE 支持、`authorization_code`+`refresh_token` grant。API scope（read:patient…write:prescription）挂在 audience `https://api.neutron.health` 上，靠 `audience` 参数请求。docs 明确：`write:prescription` 只发给被授权的 prescriber——所以登录的必须是有开方权限的沙箱医生账号。
- ✅ **treatment 选择启发式**已实现（`prescription._rank_candidates`，优先纯仿制药；搜 20 条因为目录把单方埋在复方后面）。
- ⏳ **`dispenseUnit` 合法取值**：现在按剂型默认 Each/Milliliter/Gram 且可改；真开方时若 Photon 报非法值，按 502 错误信息调整。
- ⏳ **provider token 的开方人归属**：应归登录医生（createPrescription 无 prescriberId 参数，从认证上下文取）——交互测试时在 Neutron 后台确认。
- 🛑 **唯一真正的卡点见关键发现 #7：SPA client id 错误**。

## ⏸️ 已搁置（明确不做）
- 管制药 / EPCS（等真实 prescriber）。
- 真实生产开药（等 prescriber + Photon 生产开通 + BAA）。
- Webhook 实时订单状态回流（需后端公网可达 / 内网穿透）。
- 语音录入（fast-follow）。

---

## 怎么跑
```bash
# 后端
cd /Users/alphonsesun/Desktop/codeworks/SouforgeTechCodeworks/Osler
.venv/bin/python demo/demo_app.py        # 读 demo/.env，打印 photon=on (sandbox)，127.0.0.1:5000

# 手机
cd mobile && npx expo start              # 按 i 进 iOS 模拟器
```
快速验证后端+Photon（只读/同步，安全）：
```bash
PYTHONPATH=demo .venv/bin/python -c "import demo_app, photon_client as PH; print('token', bool(PH.get_token())); print(PH.search_medications('omeprazole',3))"
```
看 authorize URL 当下能否被 Auth0 接受（**现在会 400 Unknown client，修好 SPA id 后应进登录页**）：
```bash
PYTHONPATH=demo .venv/bin/python -c "
import demo_app, photon_oauth as O, urllib.request as u, urllib.error as e
url=O.build_authorize_url('oslianrx://photon-connected')
try: b=u.urlopen(url,timeout=20).read().decode('utf8','replace')
except e.HTTPError as x: b=x.read().decode('utf8','replace')
import re,html; print(re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',b)))[:400])"
```

## 怎么测（端到端，要你的 Neutron 登录）
**前置已就绪**：SPA client id 已修对、`http://127.0.0.1:5000` 已在白名单、后台无需再动。直接测：
1. 起后端（banner 应是 `photon=on (sandbox) provider_oauth=ready`）+ `npx expo start` 按 `i` 进模拟器。
2. **Settings → PHOTON · E-PRESCRIBING → Connect Photon** → 系统浏览器开 Neutron 登录 → 用**有开方权限的沙箱 provider 账号**登录 → 自动回 app，卡片显示 "Connected · 医生名"。
3. Patients 选/分析一个案例（如 GERD）→ Reasoning → 某药 **Prescribe** → 原生审核屏。
4. 选药（默认已选纯仿制药）、改 sig/数量/天数 → **Sign & Send** → 回 Rx tab 看到 sent 记录。
5. 去 `app.neutron.health` 后台确认 prescription/order 真的建了、开方人=登录医生。
- 失败排查：Connect 卡在浏览器报错 → SPA id / 回调白名单；Sign 报 401 → 没连或 token 过期（重连）；Sign 报 502 → Photon GraphQL 拒绝，错误信息在 app 弹窗 + 后端日志里，多半是某药 treatment 不可开或 dispenseUnit 非法，换个药或调 `_CREATE_RX`。

## 凭证 / 配置
- Photon 沙箱凭证在 `demo/.env`（gitignore）：`PHOTON_ENV=sandbox`、M2M `PHOTON_CLIENT_ID/SECRET`（✅）、`PHOTON_SPA_CLIENT_ID=0ltrnfbI0r9ku2hwR89TQkmOrSyHAEwI`（✅ 已修对，注意第 8、32 位是**大写 I** 不是小写 l）、`PHOTON_ORG_ID=org_nkLTa9shDtFLVuyP`（org `oslian-llc`）。
- 回调用裸 origin `http://127.0.0.1:5000`（已在 Photon SPA Whitelisted URLs 里），Flask 根路由接 `?code&state`。真机/局域网：把 `http://<LAN_IP>:5000` 加进 Whitelisted URLs 并设环境变量 `PHOTON_OAUTH_REDIRECT=http://<LAN_IP>:5000`。
- Neutron 后台 SPA "Oslian LLC Frontend" 的 Whitelisted URLs 现有：`http://localhost:5000, http://localhost:3000, http://127.0.0.1:5000`（裸 origin、无 query）。

## ⚠️ 未提交
本 session（2026-06-18）改动尚未 commit。改：`demo/{photon_client.py,prescription.py,demo_app.py}`、`mobile/src/{api/osler.ts,app/settings.tsx,app/prescribe.tsx}`；新增：`demo/photon_oauth.py`、本文件更新。`mobile/package-lock.json` 有一处无关改动。可整理成一个 commit（`feat(rx): native provider-OAuth Sign & Send`）。
