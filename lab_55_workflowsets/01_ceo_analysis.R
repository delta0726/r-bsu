# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : ROBYN SETUP
# Chapter     : LAB 55: WORKFLOWSETS
# Module      : 01_ceo_analysis.R
# Update Date : 2021/8/17
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - {workflowset}を使って複数レシピと複数モデルを一括学習する
#   --- リサンプリング予測やチューニングでの使用を想定


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 株価リターン類似度分析
# 3 学習用データの完成
# 4 複数レシピの定義
# 5 複数モデルの定義
# 6 ワークフローセットへの登録
# 7 ワークフローセットの学習
# 8 学習結果の評価
# 9 最終モデルの選定＆再学習
# 10 最終モデルの評価


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
library(tidyverse)
library(tidymodels)
library(workflowsets)
library(finetune)
library(vip)
library(parallel)
library(doParallel)
library(job)
library(DataExplorer)
library(janitor)
library(timetk)
library(lubridate)
library(tidyquant)
library(tictoc)


# 1 データ準備 --------------------------------------------------------------------------------

# CEOデータ --------------------------------------------------

# データロード
ceo_compensation_raw_tbl <- read_csv("lab_55_workflowsets/00_data/ceo_compensation_2008.csv")

# データ確認
ceo_compensation_raw_tbl %>% print()
ceo_compensation_raw_tbl %>% glimpse()

# 欠損値確認
ceo_compensation_raw_tbl %>% 
  profile_missing() %>%
  arrange(desc(pct_missing)) 

# データ加工
ceo_compensation_tbl <- 
  ceo_compensation_raw_tbl %>%
    clean_names() %>% 
    mutate(x6_year_annual_total_return = parse_number(x6_year_annual_total_return)) %>%
    mutate(total_return_during_tenure = parse_number(total_return_during_tenure)) %>%
    select(-starts_with("x2008_"), -x5_year_compensation_total, -x6_year_average_compensation)

# データ確認
ceo_compensation_tbl %>% print()
ceo_compensation_tbl %>% glimpse()


# 株価データ ------------------------------------------------

# データロード
stock_prices_tbl <- 
  read_rds("lab_55_workflowsets/00_data/stock_prices.rds") %>%
  distinct()

# 時系列プロット作成
stock_prices_tbl %>%
  filter(symbol == "AAPL") %>%
  plot_time_series(date, adjusted)
  
# データ加工
# --- 年次リターンに変換
performance_by_year_tbl <- 
  stock_prices_tbl %>%
    group_by(symbol) %>%
    summarize_by_time(.by = "year", 
                      n = n(), 
                      total_performance = (last(adjusted) - first(adjusted) ) / first(adjusted)) %>%
    ungroup() %>%
    mutate(year = str_glue("performance_{year(date)}")) %>%
    select(symbol, year, total_performance) %>%
    pivot_wider(id_cols     = symbol, 
                names_from  = year, 
                values_from = total_performance) %>%
    rowwise() %>%
    mutate(performance_mean    = mean(c_across(starts_with("perf")), na.rm = T), 
           performance_median  = median(c_across(starts_with("perf")), na.rm = T), 
           performance_stdev   = sd(c_across(starts_with("perf")), na.rm = T), 
           performance_missing = sum(is.na(c_across(starts_with("perf"))), na.rm = TRUE)) %>%
    ungroup()

performance_by_year_tbl %>% glimpse()


# 2 株価リターン類似度分析 -------------------------------------------------------------

# データ加工
# --- 株価をリターンに変換
# --- ワイド型に変換して列方向に銘柄を並べる
stock_returns_wide_tbl <- 
  stock_prices_tbl %>%
    select(symbol, date, adjusted) %>%
    group_by(symbol) %>%
    mutate(daily_returns = diff_vec(adjusted, lag = 1, silent = TRUE) / lag_vec(adjusted, lag = 1)) %>%
    ungroup() %>%
    pivot_wider(id_cols     = date, 
                names_from  = symbol, 
                values_from = daily_returns, 
                values_fill = 0) %>%
    drop_na() %>%
    select(-date)

# 相関係数行列
stock_returns_wide_tbl %>%
  cor() %>%
  as_tibble()

# クラスタリング
# --- k-means
kmeans_obj <- 
  stock_returns_wide_tbl %>% 
    as.matrix() %>%
    t() %>%
    scale() %>%
    kmeans(centers = 10, iter.max = 20)

# クラスタデータの格納
# --- 銘柄ごとのクラスターグループ
stock_similarity_tbl <- 
  kmeans_obj$cluster %>% 
    enframe(name = "symbol", value = "kmeans_group")


# 3 学習用データの完成 ---------------------------------------------------------------

# データ結合
ceo_compensation_joined_tbl <- 
  ceo_compensation_tbl %>%
    left_join(performance_by_year_tbl, by = c("ticker" = "symbol")) %>%
    left_join(stock_similarity_tbl, by = c("ticker" = "symbol")) %>%
    filter(!is.na(total_2008_compensation))

# データ確認
ceo_compensation_joined_tbl %>% print()
ceo_compensation_joined_tbl %>% glimpse()

# 欠損値確認
# --- NAが多い（レシピを用いてNAを削除することで対応）
ceo_compensation_joined_tbl %>%
  profile_missing() %>%
  arrange(desc(pct_missing))

# リサンプリングデータの作成
set.seed(123)
resample_spec <- 
  ceo_compensation_joined_tbl %>%
  vfold_cv(v = 5)


# 4 複数レシピの定義 --------------------------------------------------------------------

# ＜ポイント＞
# - ベースレシピを元に3パターンのレシピを作成


# ベースレシピ -----------------------------------

# レシピ定義
recipe_base <- recipe(total_2008_compensation ~ ., data = ceo_compensation_joined_tbl) %>%
  step_rm(ceo, company, ticker) %>%
  step_mutate_at(kmeans_group, fn = factor) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) 

# データ確認
recipe_base %>% prep() %>% juice() %>% glimpse()


# レシピ1：NAのない特徴量のみ選択 -------------------

# レシピ定義
recipe_no_missing <- 
  recipe_base %>%
    step_select(where(function (x) !any(is.na(x))))

# データ確認
recipe_no_missing %>% prep() %>% juice() %>% glimpse()


# レシピ2：NAを平均値で補完 ------------------------

# レシピ定義
recipe_impute_mean <- 
  recipe_base %>%
    step_impute_mean(all_predictors())

# データ確認
recipe_impute_mean %>% prep() %>% juice() %>% glimpse()


# レシピ3：NAをknnアルゴリズムで補完 ---------------

# レシピ定義
# --- 要チューニング
recipe_impute_knn <- 
  recipe_base %>%
    step_impute_knn(all_predictors(), neighbors = tune())

# データ確認
# --- チューニングが必要なためデータ確認はできない
#recipe_impute_knn %>% prep() %>% juice() %>% glimpse()



# 5 複数モデルの定義 -------------------------------------------------------------------

# ＜ポイント＞
# - 3パターンのモデルを作成


# モデル1
# --- Elastic Net
# --- 要チューニング
glmnet_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

# モデル2
# --- XGBoost
# --- 要チューニング
xgboost_spec <- 
  boost_tree(mode = "regression", 
             trees = 500, 
             learn_rate = tune(), 
             min_n      = tune()) %>%
  set_engine("xgboost")

# モデル3
# --- SVM
# --- 要チューニング
svm_spec <- 
  svm_rbf(mode      = "regression", 
          cost      = tune(), 
          rbf_sigma = tune(), 
          margin    = tune()) %>%
  set_engine("kernlab")



# 6 ワークフローセットの作成 -------------------------------------------------------------------


# ワークフローセットの作成
# --- 9パターン(3レシピ * 3モデル)
# --- wflow_idはリストで設定した名称を結合して作成される
wflwset_setup <- 
  workflow_set(preproc = list(no_missing  = recipe_no_missing,　
                              impute_mean = recipe_impute_mean,
                              impute_knn  = recipe_impute_knn),
               models = list(glmnet  = glmnet_spec, 
                             xgboost = xgboost_spec, 
                             svm     = svm_spec), 
               cross = TRUE) 

# 確認
wflwset_setup %>% print()

# 構造確認
# --- infoの中にワークフローが格納されている
# --- レシピやモデルのオブジェクトは直接格納されていない（名前だけ表示）
wflwset_setup$info[[1]]
wflwset_setup$info[[1]]$workflow
wflwset_setup$info[[1]]$preproc
wflwset_setup$info[[1]]$model


# 7 ワークフローセットの学習 -------------------------------------------------------------------

# 並列処理の設定
cores <- parallel::detectCores(logical = FALSE)
clusters <- makePSOCKcluster(cores)
registerDoParallel(clusters)
tic()

# Long running script
# - Run as Job: See Addins > Run Selection as Job

set.seed(123)
wflwset_tune_results <- 
  wflwset_setup %>%
    workflow_map(fn        = "tune_race_anova", 
                 resamples = resample_spec, 
                 grid      = 15, 
                 metrics   = metric_set(rmse, rsq), 
                 verbose   = TRUE)

# 並列処理の終了
doParallel::stopImplicitCluster()
toc()


# 確認
wflwset_tune_results %>% print()
wflwset_tune_results$option[[1]] %>% glimpse()
wflwset_tune_results$result[[1]] %>% glimpse()


# 8 学習結果の評価 -------------------------------------------------------------------

# ＜ポイント＞
# - プロットを作成するとRが落ちるのでプロット作成を中止
#   --- 0.1.0ではバグ修正が告知されている


# プロット確認
# wflwset_tune_results %>% 
#   autoplot() + 
#     scale_color_tq() +
#     theme_tq() 
  
# モデル精度の順位
# --- RMSE
wflwset_tune_results %>% 
  rank_results(rank_metric = "rmse") %>%
  filter(.metric == "rmse") %>% 
  print(n = nrow(.))

# プロット作成
# --- ベストパラメータ
# wflwset_tune_results %>% autoplot(id = "impute_mean_xgboost", metric = "rmse")
# wflwset_tune_results %>% autoplot(id = "impute_knn_xgboost", metric = "rmse")


# 9 最終モデルの選定＆再学習 ----------------------------------------------------------

# 最良パラメータの抽出
# --- RMSE基準
params_best_model <- 
  wflwset_tune_results %>%
  extract_workflow_set_result(id = "impute_mean_xgboost") %>% 
    select_best(metric = "rmse")

# 最終モデルので学習
wflw_fit_final <- 
  wflwset_tune_results %>%
    pull_workflow("impute_mean_xgboost") %>%
    finalize_workflow(params_best_model) %>%
    fit(ceo_compensation_joined_tbl)

# 予測値の作成
predictions_tbl <- 
  wflw_fit_final %>%
    predict(new_data = ceo_compensation_joined_tbl) %>%
    bind_cols(ceo_compensation_joined_tbl) %>%
    select(.pred, total_2008_compensation, everything())


# 10 最終モデルの評価 -----------------------------------------------------------------

# ＜ポイント＞
# - プロットを作成するとRが落ちるのでプロット作成を中止
#   --- 0.1.0ではバグ修正が告知されている


# プロット作成
# --- 予測
# predictions_tbl %>%
#   ggplot(aes(total_2008_compensation, .pred)) +
#   geom_point() +
#   geom_abline(slope = 1, intercept = 0, color = "blue")

# 変数重要度分析
importance_tbl <- wflw_fit_final$fit$fit$fit %>% vip::vi()
importance_tbl %>% print()
