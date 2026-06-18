# Osler·Rx 电子处方整合 — HANDOFF

> 给下一个 session 接力用。读这份 + 代码就能继续，无需上文。
> 仓库：`SouforgeTechCodeworks/Osler`，分支 **`feature/mobile-demo`**（不是 SoulLink 目录）。
> 详细背景计划：`~/.claude/plans/hi-claude-code-alphonse-oslian-app-anita-cozy-leaf.md`

## 一句话状态
"症状→符号引擎荐药→开处方→送药房" 的闭环，**后端↔Photon 沙箱已实跑通、app 全流程已在 iOS 模拟器跑起来**；下一步是把"开处方"那段从"跳浏览器"改成**原生一屏 Sign & Send**（非管制药）。

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

---

## 🔑 关键发现 / 架构定调（别再重新踩）
1. **M2M 令牌不能建处方**。当前 M2M scope = `read:patient write:patient read:prescription read:order write:order`，**没有 `write:prescription`**（Photon 故意如此——开处方必须绑认证过的医生身份，不让后端机器代开）。`createPrescription` 实跑报 `MISSING_PERMISSIONS: write:prescription`。
   → **结论架构：医生一次性 OAuth 登录（系统浏览器）→ app 拿到 provider 用户令牌 → 用该令牌原生调 `createPrescription`+`createOrder`。浏览器只用于偶尔登录，开方全程原生、不跳。**
2. **嵌入式 WebView 登录此路不通**：Auth0/Google 封禁嵌入 WebView（试过，报 Neutron "something went wrong"）。**系统浏览器**登录正常（已验证 Photon 登录页能在系统浏览器加载）。
3. **没有真实持照/带 DEA 号的开方医生**（团队现状）→ **现在任何真实开药都做不了（不止管制药）**，只能 sandbox 演示。真上线（任何药）卡在"招到一个 prescriber"这个商务步骤。
4. **管制药/EPCS：搁置**。需真实医生 + Experian 身份核验(IAL2) + 2FA 令牌 + DEA 第三方软件审计（或把签名这步路由到 Photon 已认证流程=必然"跳"）。架构上：检测 `Medication.controlled` → 走 Photon 认证 2FA 流程。**现在不做**。
5. 杂项坑：① 搜药别请求 `controlled` 字段（沙箱有 null 脏数据破坏 non-null 查询）；`Medication.id` 即 `treatmentId`。② `createOrder` 省略 `pharmacyId` → Photon 自动路由；`address` 用于路由。③ patient `phone` 必须合法 NANP 区号（`555` 被拒，用 `+12025550102`）。④ 内存缓存：analyze 与 prescribe 须在同一次后端运行内；app 整页 reload 清空患者→需重新分析。⑤ AirPlay 占 :5000 → 用 `127.0.0.1` 非 localhost（已配）。
6. **写操作会被 auto-mode 安全分类器拦**：直接用 Bash 跑 `createPrescription`/`createOrder` 会被拦（写外部系统）。要么经 app 后端在 UI 操作时触发，要么先取得用户明确授权。

---

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

---

## ❓待解决的技术问题（下一个 session 先确认）
- Photon SPA OAuth 的精确参数：authorize/token 端点、PKCE、redirect scheme、scope 列表是否含 `write:prescription`（可能要在 Neutron 后台给 SPA app 勾选/或用户登录时授予）。参考 Photon Elements/Authentication 文档 + 可对 `https://docs.photon.health/docs/authentication.md` 再查。
- createPrescription 用 provider token 后，开方人是否就归该登录医生（应是）。
- `dispenseUnit` 合法取值；treatment 选择启发式（优先纯仿制药而非复方）。

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

## 凭证 / 配置
- Photon 沙箱凭证在 `demo/.env`（gitignore，已配好）：`PHOTON_ENV=sandbox`、M2M `PHOTON_CLIENT_ID/SECRET`、`PHOTON_SPA_CLIENT_ID`、`PHOTON_ORG_ID=org_nkLTa9shDtFLVuyP`（org `oslian-llc`）。
- Neutron 后台：`app.neutron.health`（你的账号 = sandbox 测试 provider）。SPA 白名单已含 `http://localhost:5000`、`http://127.0.0.1:5000`（真机再加局域网 IP）。

## ⚠️ 未提交
本 session 所有改动**尚未 commit**（`git -C Osler status` 可见：改 `demo/demo_app.py`、`mobile/src/{api/osler.ts,app/_layout.tsx,app/reasoning.tsx,app/prescribe.tsx,components/DrugCard.tsx}`；新增 `demo/{photon_client.py,prescription.py,prescribe_embed.html,.env,.env.example}`、`mobile/src/app/prescriptions.tsx`、`mobile/src/components/PrescriptionCard.tsx`、本文件）。新 session 可先决定是否整理成 commit。
