# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Marketing Analytics with R & Python
# Chapter     : Lab 58: Customer Lifetime Value (CLV) with R Shiny
# Module      : lab_58_cutomer_ltv.R
# Update Date : 2021/6/27
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 1 準備
# 2 コホート分析
# 3 機械学習
# 3.1 データ分割 (2-Stages)
# 3.2 FEATURE ENGINEERING (RFM)
# 3.3 レシピ作成
# 3.4 モデル構築
# 3.5 テストデータを用いた評価
# 3.6 変数重要度分析
# 3.7 分析プロセスの保存
# 4 検証課題へのアプローチ


# 1 準備 ------------------------------------------------------------------------

# ライブラリ
library(tidymodels)
library(vip)
library(tidyverse)
library(timetk)
library(lubridate)


# データロード
# --- テキストファイルを読込
cdnow_raw_tbl <-
  vroom::vroom(file = "lab_58_cust_lifetime_r/data/CDNOW_master.txt",
               delim = " ",
               col_names = FALSE)

# データ加工
cdnow_tbl <-
  cdnow_raw_tbl %>%
    select(X2, X3, X5, X8) %>%
    set_names(c("customer_id", "date", "quantity", "price")) %>%
    mutate(date = ymd(as.character(date))) %>%
    drop_na()

# データ確認
cdnow_tbl %>% print()
cdnow_tbl %>% glimpse()


# 2 コホート分析 ---------------------------------------------------------------

# * 準備 ------------------------------------------------

# 初回購入日の取得
# --- カスタマーIDごと
cdnow_first_purchase_tbl <-
  cdnow_tbl %>%
    group_by(customer_id) %>%
    slice_min(date) %>%
    ungroup()

# 期間の確認
# --- 初回購入日
cdnow_first_purchase_tbl %>% pull(date) %>% range()


# * コホート期間の設定 ----------------------------------------

# ＜ポイント＞
# - 分析対象期間を1997-01-01 to 1997-03-31と設定する


# ID抽出
# --- 対象期間に購入したカスタマーID
ids_in_cohort <-
  cdnow_first_purchase_tbl %>%
    filter_by_time(.date = date, .start_date = "1997-01", .end_date   = "1997-03") %>%
    distinct(customer_id) %>%
    pull(customer_id)

# データ抽出
# --- 対象期間に購入したIDのみ
cdnow_cohort_tbl <-
  cdnow_tbl %>%
    filter(customer_id %in% ids_in_cohort)

# 期間抽出
# --- 対象期間に購入したIDで抽出している
# --- データ期間自体を1997-01-01 to 1997-03-31としているわけではない点に注意
cdnow_cohort_tbl %>% tk_index() %>% tk_get_timeseries_summary() %>% select(1:4)


# * 可視化: コホートの売上高 --------------------------------

# プロット作成
# --- 期間ごとの売上高
cdnow_cohort_tbl %>%
  summarize_by_time(total_price = sum(price, na.rm = TRUE),
                    .by   = "month") %>%
  plot_time_series(date, total_price, .y_intercept = 0)


# * 可視化: 個人の売上高 ------------------------------------

n    <- 1:10
ids  <- cdnow_cohort_tbl$customer_id %>% unique() %>% .[n]

# プロット作成
# --- 個別IDごとにファセットを作成
# --- なぜかgeom_point()が作動しないので削除
cdnow_cohort_tbl %>%
  filter(customer_id %in% ids) %>%
  group_by(customer_id) %>%
  plot_time_series(date, price,
                   .y_intercept = 0,
                   .smooth      = FALSE,
                   .facet_ncol  = 2,
                   #.interactive = FALSE,
                   .title = "Customer Purchase Behavior")


# 3 機械学習 ----------------------------------------------------------------------------

#  Frame the problem:
#  - What will the customers spend in the next 90-Days? (Regression)
#  - What is the probability of a customer to make a purchase in next 90-days? (Classification)


# 3.1 データ分割 (2-Stages) -----------------------------------------

# ＜ポイント＞
# - カスタマーIDごとに訓練データ/テストデータのどちらに入るか決定した上で時系列データ分割


# ** Stage 1: カスタマーIDでランダムにデータ分割 ----

# ID抽出
# --- 訓練データ/テストデータの決定
set.seed(123)
ids_train <-
  cdnow_cohort_tbl %>%
    pull(customer_id) %>%
    unique() %>%
    sample(size = round(0.8 * length(.))) %>%
    sort()

# データ抽出
# --- 訓練データ
split_1_train_tbl <-
  cdnow_cohort_tbl %>%
    filter(customer_id %in% ids_train)

# データ抽出
# --- テストデータ
split_1_test_tbl  <-
  cdnow_cohort_tbl %>%
    filter(!customer_id %in% ids_train)


# ** Stage 2: 時系列データ分割 --------------------

# 時系列データ分割
# --- 訓練データ
splits_2_train <-
  split_1_train_tbl %>%
    time_series_split(date_var   = date,
                      assess     = "90 days",
                      cumulative = TRUE)

# 時系列データ分割
# --- テストデータ
splits_2_test <-
  split_1_test_tbl %>%
    time_series_split(assess     = "90 days",
                      cumulative = TRUE)

# データ分割イメージ
# --- 訓練データ
splits_2_train %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, price)

# データ分割イメージ
# --- テストデータ
splits_2_test %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, price)


# 3.2 FEATURE ENGINEERING (RFM) ------------------------------------
#   - Most challenging part
#   - 2-Stage Process
#   - Need to frame the problem
#   - Need to think about what features to include


# データ集計
# --- テスト期間(90日)の合計金額
# --- テスト期間(90日)のフラグ
# --- 訓練データのテスト期間（in-sample）
targets_train_tbl <-
  splits_2_train %>%
    testing() %>%
    group_by(customer_id) %>%
    summarise(spend_90_total = sum(price),
              spend_90_flag    = 1)

# データ集計
# --- テスト期間(90日)の合計金額
# --- テストデータのテスト期間（out-sample）
targets_test_tbl <-
  splits_2_test %>%
    testing() %>%
    group_by(customer_id) %>%
    summarise(spend_90_total = sum(price),
              spend_90_flag    = 1)


# ** Make Training Data ----
#    - What features to include?
#    - RFM: Recency, Frequency, Monetary

# 日付取得
# --- 訓練データの最終日
max_date_train <-
  splits_2_train %>%
    training() %>%
    pull(date) %>%
    max()

# 訓練データの作成
# --- recency： 最近購入した日からの経過日数

# --- frequency： 出現頻度
train_tbl <-
  splits_2_train %>%
    training() %>%
    group_by(customer_id) %>%
    summarise(recency    = (max(date) - max_date_train) / ddays(1),
              frequency  = n(),
              price_sum  = sum(price, na.rm = TRUE),
              price_mean = mean(price, na.rm = TRUE)) %>%
    left_join(targets_train_tbl) %>%
    replace_na(replace = list(spend_90_total = 0,
                              spend_90_flag  = 0)) %>%
    mutate(spend_90_flag = as.factor(spend_90_flag))

# テストデータの作成
test_tbl <-
  splits_2_test %>%
    training() %>%
    group_by(customer_id) %>%
    summarise(recency     = (max(date) - max_date_train) / ddays(1),
              frequency   = n(),
              price_sum   = sum(price, na.rm = TRUE),
              price_mean  = mean(price, na.rm = TRUE)) %>%
    left_join(targets_test_tbl) %>%
    replace_na(replace = list(spend_90_total = 0,
                              spend_90_flag  = 0)) %>%
    mutate(spend_90_flag = as.factor(spend_90_flag))

# データ確認
train_tbl %>% print()
test_tbl %>% print()


# 3.3 レシピ作成 ------------------------------------

# レシピ作成
# --- 回帰問題用
recipe_spend_total <-
  recipe(spend_90_total ~ ., data = train_tbl) %>%
    step_rm(spend_90_flag, customer_id)

# レシピ作成
# --- 分類問題用
recipe_spend_prob <-
  recipe(spend_90_flag ~ ., data = train_tbl) %>%
    step_rm(spend_90_total, customer_id)


# レシピ確認
# --- 回帰問題用
recipe_spend_total %>% prep() %>% juice() %>% glimpse()
recipe_spend_total %>% summary()

# レシピ確認
# --- 分類問題用
recipe_spend_prob %>% prep() %>% juice() %>% glimpse()
recipe_spend_prob %>% summary()


# 3.4 モデル構築 ----------------------------------------

# モデル1：回帰問題
wflw_spend_total_xgb <-
  workflow() %>%
    add_model(boost_tree(mode = "regression") %>% set_engine("xgboost")) %>%
    add_recipe(recipe_spend_total) %>%
    fit(train_tbl)

# モデル2：分類問題
wflw_spend_prob_xgb <-
  workflow() %>%
    add_model(boost_tree(mode = "classification") %>% set_engine("xgboost")) %>%
    add_recipe(recipe_spend_prob) %>%
    fit(train_tbl)


# 3.5 テストデータを用いた評価 --------------------------

# ＜ポイント＞
# - テストデータを用いてモデル評価を行う


# * 予測値の作成(テストデータ) --------------------

# 予測結果（回帰モデル）
# --- spend_90_total（90日の合計金額）
pred_reg <-
  wflw_spend_total_xgb %>%
    predict(test_tbl) %>%
    rename(.pred_total = .pred)

# 予測結果（分類モデル）
# --- spend_90_total（90日の購入フラグ）
# --- 1となるクラス確率を算出
pred_cls <-
  wflw_spend_prob_xgb %>%
    predict(test_tbl, type = "prob") %>%
    select(.pred_1) %>%
    rename(.pred_prob = .pred_1)

# データ結合
predictions_test_tbl <-
  pred_reg %>%
    bind_cols(pred_cls, test_tbl) %>%
    select(starts_with(".pred"), starts_with("spend_"), everything())

# データ確認
predictions_test_tbl %>% glimpse()


# * モデル精度の評価(テストデータ) ------------

# 回帰モデルの評価
# --- MAE
predictions_test_tbl %>%
    mae(spend_90_total, .pred_total)

# 分類モデルの評価
# --- AUC
predictions_test_tbl %>%
  roc_auc(spend_90_flag, .pred_prob, event_level = "second")

# 分類モデルの評価
# --- ROCカーブ
predictions_test_tbl %>%
  roc_curve(spend_90_flag, .pred_prob, event_level = "second") %>%
  autoplot()


# 3.6 変数重要度分析 ----------------------------------------------------

# 分類モデル
wflw_spend_prob_xgb$fit$fit %>% vip()

# 回帰モデル
wflw_spend_total_xgb$fit$fit %>% vip()


# 3.7 分析プロセスの保存 ----------------------------------------------------

# # ディレクトリ作成
# fs::dir_create("artifacts")
#
# # ワークフロー保存
# wflw_spend_prob_xgb %>% write_rds("artifacts/model_prob.rds")
# wflw_spend_total_xgb %>% write_rds("artifacts/model_spend.rds")
#
# # 変数重要度の保存
# wflw_spend_prob_xgb$fit$fit %>% vi_model() %>% write_rds("artifacts/vi_prob.rds")
# wflw_spend_total_xgb$fit$fit %>% vi_model() %>% write_rds("artifacts/vi_spend.rds")
#
# # データ保存
# all_tbl <- bind_rows(train_tbl, test_tbl)
# predictions_all_tbl <-
#   bind_cols(
#         predict(wflw_spend_total_xgb, all_tbl) %>%
#             rename(.pred_total = .pred),
#         predict(wflw_spend_prob_xgb, all_tbl, type = "prob") %>%
#             select(.pred_1) %>%
#             rename(.pred_prob = .pred_1)
#     ) %>%
#     bind_cols(all_tbl) %>%
#     select(starts_with(".pred"), starts_with("spend_"), everything())
#
# #
# predictions_all_tbl %>% write_rds("artifacts/predictions_all_tbl.rds")


# 4 検証課題へのアプローチ --------------------------------

# ** Which customers have the highest spend probability in next 90-days? ----
#    - Target for new products similar to what they have purchased in the past

# 質問1：次の90日に最も高い消費が期待できる顧客はどれか
# --- 将来の購買行動は過去の購買行動と類似するという考えのもと、クラス確率の高いIDに期待する
predictions_test_tbl %>%
  arrange(desc(.pred_prob))

# 質問2：最近購入したのに購入が期待できない顧客はどれか
# --- クラス確率とRecencyで絞り込む
# --- 割引など購入を刺激する施策の対象とする
predictions_test_tbl %>%
  filter(recency    > -90, .pred_prob < 0.2) %>%
  arrange(.pred_prob)

# 質問3：過去の消費額が多いのに、最近の購入が少ない顧客はどれか（機会損失）
# --- 予測金額(回帰)とspend_90_totalで絞り込む
#    - Send bundle offers
predictions_test_tbl %>%
  arrange(desc(.pred_total)) %>%
  filter(spend_90_total == 0)
