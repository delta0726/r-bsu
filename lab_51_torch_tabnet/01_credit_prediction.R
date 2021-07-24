# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Tidymodels Ecosystem
# Chapter     : LAB 51: Deep Learning with Torch & Tabnet
# Module      : 01_credit_prediction.R
# Update Date : 2021/7/22
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜テーマ＞
# - {tabnet}を使って{torch}のディープラーニングを体験する
# - lending_clubデータの分類問題を解く


# ＜目次＞
# 0 準備
# 1 データ分割
# 2 特徴量エンジニアリング
# 3 モデリング
# 4 モデル精度の検証
# 5 クロスバリデーション
# 6 変数重要度分析


# 0 準備 -----------------------------------------------------------------------

# Core
library(skimr)
library(timetk)
library(tidyverse)
library(magrittr)
library(tictoc)

# Machine Learning
library(tabnet)
library(xgboost)
library(tidymodels)
library(vip)


# データロード
data("lending_club", package = "modeldata")

# データ確認
lending_club %>% glimpse()
lending_club %>% skim()


# 1 データ分割 ------------------------------------------------------------------

# データ分割
# --- 層別サンプリング： Class
set.seed(123)
splits <- lending_club %>% initial_split(strata = Class, prop = 0.80)

# データ確認
# --- 訓練データとテストデータでbad/goodの比率が同じ
splits %>% print()
splits %>% training() %>% use_series(Class) %>% table() %>% prop.table()
splits %>% testing() %>% use_series(Class) %>% table() %>% prop.table()


# 2 特徴量エンジニアリング --------------------------------------------------------

# レシピ作成
# --- tabnetモデル用
recipe_tabnet <-
  recipe(Class ~ ., training(splits)) %>%
    step_normalize(all_numeric())

# レシピ作成
# --- xgboostモデル用
recipe_xgboost <-
  recipe(Class ~ ., training(splits)) %>%
    step_dummy(all_nominal(), -Class, one_hot = TRUE)

# データ確認
recipe_tabnet %>% prep() %>% juice() %>% glimpse()
recipe_xgboost %>% prep() %>% juice() %>% glimpse()


# 3 モデリング ------------------------------------------------------------------

# * TabNet --------------------------------------------

# モデル構築
model_tabnet <-
  tabnet(mode               = 'classification',
         batch_size         = 128,
         virtual_batch_size = 128,
         epochs             = 10) %>%
    set_engine("torch", verbose = TRUE)


# ワークフロー設定＆学習
# --- 152.12 sec elapsed
tic()
wflw_tabnet_fit <-
  workflow() %>%
    add_model(model_tabnet) %>%
    add_recipe(recipe_tabnet) %>%
    fit(training(splits))
toc()


# * Saving/Loading ----

# # Serialization
# wflw_tabnet_fit %>% write_rds( "lab_51_torch_tabnet/models_01_credit/wflw_tabnet_fit.rds")
# wflw_tabnet_fit$fit$fit$fit$fit$network %>% torch::torch_save( "lab_51_torch_tabnet/models_01_credit//torch_network")

# Loading
# wflw_tabnet_fit <- read_rds("lab_51_torch_tabnet/models_01_credit//wflw_tabnet_fit.rds")
# torch_network   <- torch::torch_load("lab_51_torch_tabnet/models_01_credit/torch_network")
# wflw_tabnet_fit$fit$fit$fit$fit$network <- torch_network
# wflw_tabnet_fit


# * XGBoost ----------------------------------------

# モデル構築
model_xgboost <-
  boost_tree(mode = 'classification') %>%
    set_engine("xgboost")

# ワークフロー設定＆学習
# --- 152.12 sec elapsed
tic()
wflw_xgboost_fit <-
  workflow() %>%
    add_model(model_xgboost) %>%
    add_recipe(recipe_xgboost) %>%
    fit(training(splits))
toc()


# wflw_xgboost_fit %>% write_rds("lab_51_torch_tabnet/models_01_credit/wflw_xgboost_fit.rds")


# 4 モデル精度の検証 ------------------------------------------------------------------

# OUT SAMPLE COMPARISON -----

# AUCの算出
# ---- TabNet
wflw_tabnet_fit %>%
  predict(testing(splits), type = 'prob') %>%
    bind_cols(testing(splits)) %>%
    select(Class, starts_with('.pred_')) %>%
    roc_auc(truth = Class, .pred_good, event_level = 'second')

# AUCの算出
# ----XGBoost
wflw_xgboost_fit %>%
  predict(testing(splits), type = 'prob') %>%
    bind_cols(testing(splits)) %>%
    select(Class, starts_with('.pred_')) %>%
    roc_auc(truth = Class, .pred_good, event_level = 'second')


# 5 クロスバリデーション -----------------------------------------------------------------

# * 共通 ----------------------------------------------------------

# Foldデータの作成
set.seed(123)
resample_spec <- lending_club %>% vfold_cv(v = 5, strata = Class)


# * TabNet --------------------------------------------------------

# モデル構築
# --- 以前作成したものとepochsが異なる
model_tabnet <-
  tabnet(mode               = 'classification',
         batch_size         = 128,
         virtual_batch_size = 128,
         epochs             = 30) %>%
    set_engine("torch", verbose = TRUE)

# ワークフロー設定
wf_tabnet <-
  workflow() %>%
    add_model(model_tabnet) %>%
    add_recipe(recipe_tabnet)

# クロスバリデーション
# * THIS TAKES ABOUT 30-MINUTES ON 6 CPUs *
tic()
resample_results_tabnet <- fit_resamples(
    object = wf_tabnet,
    resamples = resample_spec,
    control = control_resamples(verbose = TRUE))
toc()

# データ保存
# write_rds(resample_results_tabnet, "lab_51_torch_tabnet/models_01_credit/resample_results_tabnet.rds")
# read_rds("lab_51_torch_tabnet/models_01_credit/resample_results_tabnet.rds") %>% collect_metrics()

# モデル精度の評価
resample_results_tabnet %>% collect_metrics()


# * XGBoost ----------------------------------------------------

# モデル構築
# --- 以前作成したものとepochsが異なる
model_xgboost <-
  boost_tree(mode = 'classification') %>%
    set_engine("xgboost")

# ワークフロー設定
wf_xgboost <-
  workflow() %>%
    add_model(model_xgboost) %>%
    add_recipe(recipe_xgboost)

# クロスバリデーション
tic()
resample_results_xgboost <-
  fit_resamples(object = wf_xgboost,
                resamples = resample_spec,
                control = control_resamples(verbose = TRUE))
toc()

# Time difference of 9.165127 secs

# データ保存
# write_rds(resample_results_xgboost , "models_01_credit/resample_results_xgboost.rds")
# read_rds("models_01_credit/resample_results_xgboost.rds") %>% collect_metrics()

# モデル精度の評価
resample_results_xgboost %>% collect_metrics()


# 6 変数重要度分析 --------------------------------------------------------------------

# 変数重要度
# --- TabNet
# --- XGBoost
wflw_tabnet_fit %>% pull_workflow_fit() %>%vip()
wflw_xgboost_fit %>% pull_workflow_fit() %>% vip()


