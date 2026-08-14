# 專案工程規則

## 強制執行方式

- 開始任何分析、規劃、修改、測試或命令執行前，必須先完整閱讀本檔。
- 若工作目錄內還有更具體的 `AGENTS.md`，也必須一併閱讀並遵守。
- 除非與系統或開發者層級指令衝突，以下規則均為強制要求。

## 核心規則

1. 不保留向後相容。過時的實作、介面或格式直接刪除；不新增相容層、不寫 migration、不保留舊版 fallback。本條針對舊版相容，不得誤刪目前需求明確要求的安全防線，例如 fail-closed 或經驗證的 persistence fallback。
2. 選擇能滿足目前需求的最簡單實作。不要建立預防性抽象、沒有必要的配置層或未被目前需求使用的擴充點。
3. 系統可以有清楚且長期可維護的分層，但實作順序必須先跑通最小端到端版本，再逐層增加能力。不得為了尚未完成的複雜度拆掉已經能正確運作的流程。
4. 元件保持模組化並維持關注點分離。每個模組應有單一清楚責任，跨層依賴必須明確。
5. 優先使用成熟、持續維護且已有實務驗證的函式庫。沒有明確理由，不自行重寫既有成熟能力。
6. 新增套件或自行實作前，先檢查專案目前的依賴、標準函式庫與既有工具是否已能完成需求。不得預設現有依賴做不到。
7. 架構決策以長期維護為準。不接受「先暫時這樣，以後再全部換掉」的方案；最小實作也必須位於正確的長期架構方向上。
8. 設計新功能前，先研究成熟產品、主流函式庫或已驗證系統如何解決相同問題。優先採用已證實有效的模式，不從零發明。

## 任務開始檢查

開始動手前，確認已完成以下事項：

- 已閱讀本檔與適用範圍內的其他 `AGENTS.md`。
- 已閱讀相關程式碼、測試、依賴與現有架構。
- 已確認哪些舊實作應直接刪除，而不是包裝成相容層。
- 已找出可交付的最小端到端路徑。
- 已確認現有依賴是否足以完成需求。
- 已參考成熟實作或已驗證模式；若未採用，已能說明具體原因。
- 已確認方案同時符合目前需求與長期架構。

## 全身 JEPA 強制架構

本節適用於所有全身 factual observation/forecast 的訓練、重訓、外部驗證、checkpoint 與 runtime candidate。開始任何模型任務前，必須先完整閱讀本節。自本規則生效後，所有正式訓練一律使用下述新架構；不得以舊的 monolithic whole-body、任意 dynamic routing、單一 shared future latent 或獨立 `fit_joint` checkpoint 作為正式訓練結果。

### 唯一正式架構

```text
validated target-horizon expert cells
                ↓
           system layer
                ↕
            organ layer
                ↕
             body layer
```

- 既有且已驗證的 target × horizon 專家格子必須投影到 `system layer`，作為預測錨點，不重新發明或捨棄。
- `system layer` 表示功能系統，例如心血管灌流、呼吸交換、腎清除、酸鹼電解質、內分泌代謝、肝合成解毒、消化吸收、造血凝血與免疫發炎；這些名稱不得作為器官節點。
- `organ layer` 只允許解剖器官／結構節點：心臟、血管、肺、腎、肝、胰臟、胃、腸、甲狀腺、腎上腺、骨髓、脾、腦與骨骼肌。胃與腸、脾與淋巴功能不得再合成一個假器官。程式中的 canonical 名稱以 `osler_jepa.organ_schema.CANONICAL_ORGAN_VARIABLES` 為準。
- 系統與器官之間是明確的多對多關係，唯一合法映射是 `canonical_target_organ_membership`；例如酸鹼系統可連到腎與肺，但酸鹼本身不是器官。
- 唯一允許的跨層邊是 `system <-> organ <-> body`；器官層內只允許 canonical sparse `organ <-> organ` 邊。
- 禁止任何 `system -> body`、`body -> system` 的直接參數、模組、捷徑或 loss；資訊必須經過相鄰層。
- 人體層只能透過器官層影響系統層；系統層也只能透過器官層影響人體層。
- 器官彼此影響必須經 `canonical_organ_adjacency`，不得使用無限制的全連接 organ attention；每次新增器官邊都必須重新通過 no-organ-coupling ablation。
- 缺少專家值時使用訓練正規化空間的 population median，並保留明確 availability mask；不得使用物理 0，也不得讓 NaN 進入模型。
- residual 必須有界且 zero-initialized，使未訓練模型精確等於既有 expert anchor。

正式實作為 `osler_jepa.adjacent_hierarchy.AdjacentThreeLevelResidualAdapter`。新增訓練入口必須直接使用這個 adapter，不得複製一份相似架構，也不得新增能繞過 system、organ、body staged training 的正式入口。

### 強制訓練順序

1. 先訓練 system-local target predictor，關閉 organ/body feedback。
2. 凍結 system，訓練 `system <-> organ` 與 sparse `organ <-> organ` residual，關閉 body feedback。
3. 凍結 system 與 organ，再訓練 `organ <-> body` residual。
4. 最後獨立訓練 target-specific distributional scale head，再做 patient-cluster conformal 校準。scale head 的輸入必須從 physiology latent `detach`，校準或 measurement-process loss 不得反向污染 physiology latent 或 point prediction。
5. 比較架構時，總 epoch budget必須固定並分配到 system、organ、body 三階段，不得讓新架構多拿訓練預算。

正式預設入口為：

```bash
python teacher_anchored_joint_jepa_audit.py \
  --event-examples ... \
  --modules ...
```

命令未指定 `--variants` 時必須只訓練 `adjacent_three_level`。`whole_body_joint_jepa.py` / `fit_joint` 只負責新架構內的 JEPA backbone 與特徵抽取，不得將其單獨輸出視為完整全身候選。舊架構只能由研究者明確傳入 `--variants` 做 bounded ablation；不得成為預設、不得寫入 runtime registry、不得產生可 promotion checkpoint，也不得取代新架構 checkpoint。

### 強制驗證與輸出邊界

- 每個 target × horizon 單獨驗證，不要求整個 checkpoint 一次接管全部格子。
- 每格必須通過 7-seed patient-heldout、hospital、care-unit、time、patient bootstrap、persistence、既有 expert teacher、trained no-route ablation、conformal coverage 與 collapse/SVD gate，才可申請接管。
- 每條實際啟用的耦合路徑必須各自通過 matched ablation：direct `organ <-> organ` 必須勝過 no-organ-coupling；body-mediated `organ -> body -> organ` 必須勝過 no-body-feedback。未通過的路徑必須在該 target × horizon 關閉，不得拖累或冒充另一條已驗證路徑。
- 未通過的格子保持既有 validated expert 或 persistence；此 fallback 是安全政策，不是舊版相容層。
- 不得在 runtime 手寫固定 interval half-width。只有 exact target × horizon 的 patient-specific conformal artifact 同時通過 patient、hospital、care-unit 與 time gate，才可輸出 `lower` / `upper`；否則固定輸出 `null` 與明確的 calibration status。
- 跨器官效果只有在 trained no-route ablation 通過時才可宣稱。
- direct 器官間耦合效果只有在 trained no-organ-coupling ablation 通過時才可宣稱；經人體層中介的器官耦合效果只有在 trained no-body-feedback ablation 通過時才可宣稱。
- 所有輸出維持 factual research-only，固定 `causal_claim_allowed: false`；沒有 RCT 或可靠因果識別時不得輸出治療反事實或臨床決策宣稱。

### 病人個體狀態

- 個體化只能由 `osler_jepa.patient_state.PatientStateAdapter` 放在解剖器官層；不得直接修改 system token 或 body token。
- patient state 必須由 anchor 前的因果時間歷史做 neural predict-update，必要時可融合已驗證的機制 belief；padding 必須有顯式 mask。
- 未訓練 patient residual 必須精確等於零，使輸出保持既有 population/expert anchor。
- 每個 target × horizon 必須同時顯著勝過 population anchor 與「相同模型、相同參數量、但病人歷史和 belief 被錯配」的 capacity-matched placebo。
- 個體化區間使用 patient-weighted split conformal 與 causal-order adaptive conformal；未解出的 future observation 不得提前更新病人誤差狀態。
- latent belief 只能稱為模型內部 patient state；除非它改善 held-out 下游可觀測 target，不得宣稱它是病人的真實隱藏生理量。

### 任務開始追加檢查

- 已確認正式候選使用 `AdjacentThreeLevelResidualAdapter`。
- 已確認 `organ_names` 全部是解剖器官，未混入灌流、酸鹼、凝血、發炎等功能系統名稱。
- 已確認未傳入 `--variants` 時只會訓練 `adjacent_three_level`。
- 已確認程式中沒有 direct system/body edge。
- 已確認採用 system-first、organ-second、body-third 的 staged training。
- 已確認每個候選格的 artifact 記錄唯一已驗證耦合路徑與其 disable flags；runtime 不得重新開啟未通過的路徑。
- 已確認舊架構若出現，只是明確指定的 ablation，而非預設路徑。
- 已執行 `test_adjacent_hierarchy.py` 與相關 training/gate regression tests。
