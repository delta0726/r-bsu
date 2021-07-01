# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Tidymodels Ecosystem
# Chapter     : LAB 52: STACKS
# Module      : 01_churn_classification_stacks.R
# Update Date : 2021/7/1
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目的＞
# - 以下の３モデルをスタッキング・アンサンブルして予測を作成する
#   --- ElasticNet / Random Forest / XGBoost


# ＜参考＞
# STACKS - TIDY MODEL STACKING
# https://stacks.tidymodels.org/



# ＜目次＞
# 0 準備
# 1 データ加工
# 2 サブモデルの構築
# * Elastic Net
# * Random Forest
# * XGBoost
# 3 スタッキング・アンサンブル
# * アンサンブル対象のモデルを選択
# * スタッキングモデルのウエイト決定
# * スタッキングモデルの学習
# * 最終予測の出力


# 0 準備 ---------------------------------------------------------------------

# ライブラリ ---------------------------------------------

# Machine Learning Libraries
library(xgboost)
library(ranger)
library(glmnet)

# Tidymodels
library(tidymodels)
library(stacks)

# Parallel Processing
library(doParallel)

# Core
library(tidyverse)


# 並列処理の設定 ----------------------------------------

# コア数の取得
all_cores <- parallel::detectCores(logical = FALSE)

# 並列設定
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# parallel::stopCluster(cl)


# 1 データ加工 ------------------------------------------------------------------

# データ準備 --------------------------------------------

# データロード
customer_churn_tbl <- read_rds("lab_52_stacks/00_data/customer_churn.rds")

# データ確認
customer_churn_tbl %>% glimpse()


# 前処理 -----------------------------------------------

# レシピ作成
# --- 数値データ：基準化
# --- カテゴリカルデータ：ダミー変数に変換
recipe_spec <-
  recipe(churn ~ ., data = customer_churn_tbl) %>%
    step_normalize(all_numeric()) %>%
    step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

# データ確認
recipe_spec %>% prep() %>% juice() %>% glimpse()


# K-FOLD SPEC ----

# データ分割
# --- churnで層別サンプリング
# --- 5Foldクロスバリデーション(リサンプリング)
set.seed(123)
resamples <- customer_churn_tbl %>% vfold_cv(v = 5, strata = churn)

# データ確認
resamples %>% print()


# 2 サブモデルの構築 -----------------------------------------------------------

# ＜対象モデル＞
# * Elastic Net
# * Random Forest
# * XGBoost


# * Elastic Net ------------------------------------

# モデル構築
model_glmnet <-
  logistic_reg(mode  = "classification",
               penalty = tune(),
               mixture = tune()) %>%
    set_engine("glmnet")

# ワークフロー設定
wflw_spec_glmnet <-
  workflow() %>%
    add_model(spec = model_glmnet) %>%
    add_recipe(recipe_spec)

# チューニング
# --- Time difference of 16.76374 secs
t0 <- Sys.time()
set.seed(123)
tune_results_glmnet <-
  wflw_spec_glmnet %>%
    tune_grid(resamples  = resamples,
              grid       = 5,
              control    = control_grid(verbose       = TRUE,
                                        allow_par     = TRUE,
                                        save_pred     = TRUE,
                                        save_workflow = TRUE))
t1 <- Sys.time()
t1 - t0

# 結果確認
tune_results_glmnet %>% print()

# 結果保存
# tune_results_glmnet %>% write_rds("01_tune_results/tune_results_glmnet.rds")
# tune_results_glmnet <- read_rds("01_tune_results/tune_results_glmnet.rds")

# チューニング結果
tune_results_glmnet %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  arrange(-mean)

# オブジェクトサイズ
tune_results_glmnet %>% object.size() %>% format(units = "MB")


# * Random Forest --------------------------------

# モデル構築
model_rf <-
  rand_forest(mode  = "classification",
              mtry  = tune(),
              min_n = tune(),
              trees = 500) %>%
    set_engine("ranger")

# ワークフロー設定
wflw_spec_rf <-
  workflow() %>%
    add_model(model_rf) %>%
    add_recipe(recipe_spec)

# チューニング
# --- Time difference of 27.0812 secs
t0 <- Sys.time()
set.seed(123)
tune_results_rf <-
  wflw_spec_rf %>%
    tune_grid(resamples  = resamples,
              grid       = 5,
              control    = control_grid(verbose       = TRUE,
                                        allow_par     = TRUE,
                                        save_pred     = TRUE,
                                        save_workflow = TRUE))
t1 <- Sys.time()
t1 - t0

# 結果確認
tune_results_rf %>% print()

# 結果保存
# tune_results_rf %>% write_rds("01_tune_results/tune_results_rf.rds")
# tune_results_rf <- read_rds("01_tune_results/tune_results_rf.rds")

# チューニング結果
tune_results_rf %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  arrange(-mean)

# オブジェクトサイズ
tune_results_rf %>% object.size() %>% format(units = "MB")


# * XGBoost ----------------------------------------

# モデル構築
model_xgb <-
  boost_tree(mode       = "classification",
             min_n      = tune(),
             learn_rate = tune(),
             trees      = 500) %>%
    set_engine("xgboost")

# ワークフロー設定
wflw_spec_xgb <-
  workflow() %>%
    add_model(model_xgb) %>%
    add_recipe(recipe_spec)

# チューニング
# --- Time difference of 32.72192 secs
t0 <- Sys.time()
set.seed(123)
tune_results_xgb <-
  wflw_spec_xgb %>%
    tune_grid(resamples  = resamples,
              grid       = 5,
              control    = control_grid(verbose       = TRUE,
                                        allow_par     = TRUE,
                                        save_pred     = TRUE,
                                        save_workflow = TRUE))
t1 <- Sys.time()
t1 - t0

# 結果確認
tune_results_xgb %>% print()

# 結果保存
# tune_results_xgb %>% write_rds("01_tune_results/tune_results_xgb.rds")
# tune_results_xgb <- read_rds("01_tune_results/tune_results_xgb.rds")

# チューニング結果
tune_results_xgb %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  arrange(-mean)

# オブジェクトサイズ
tune_results_xgb %>% object.size() %>% format(units = "MB")


# 3 スタッキング・アンサンブル ----------------------------------------------------

# ＜プロセス＞
# * アンサンブル対象のモデルを選択
# * スタッキングモデルのウエイト決定
# * スタッキングモデルの学習
# * 最終予測の出力


# * アンサンブル対象のモデルを選択 -----------------------------

# モデル登録
# --- チューニング結果のオブジェクトを登録
model_stack <-
  stacks() %>%
    add_candidates(tune_results_glmnet) %>%
    add_candidates(tune_results_rf) %>%
    add_candidates(tune_results_xgb)

# 確認
model_stack %>% print()


# * スタッキングモデルのウエイト決定 ---------------------------

# スタッキングの実行
# --- モデルスタックの｢スタッキング係数｣を決定する
# --- LASSOモデルを使用するので、不要なモデルのウエイトはゼロとなる（高相関のモデルを排除）
model_stack_blend <-
  model_stack %>%
    blend_predictions(metric = metric_set(roc_auc))

# 確認
model_stack_blend %>% print()

# オブジェクトサイズ
model_stack_blend %>% object.size() %>% format(units = "MB")


# * Visualize candidate performance -----------------------

model_stack_blend %>% autoplot()
model_stack_blend %>% autoplot(type = "members")
model_stack_blend %>% autoplot(type = "weights")


# * スタッキングモデルの学習 ----------------------------------

# アンサンブルモデルの学習
model_stack_fit <-
  model_stack_blend %>%
    fit_members()

# データ確認
model_stack_fit %>% glimpse()

# オブジェクトサイズ
model_stack_fit %>% object.size() %>% format(units = "MB")


# * 最終予測の出力 ---------------------------------------------

# 予測の作成
# --- アンサンブル・モデルの予測
churn_scores_tbl <-
  model_stack_fit %>%
    predict(customer_churn_tbl, type = "prob") %>%
    bind_cols(customer_churn_tbl)

# データ確認
churn_scores_tbl %>% glimpse()

# モデル精度の確認
churn_scores_tbl %>% roc_auc(churn, .pred_yes)
