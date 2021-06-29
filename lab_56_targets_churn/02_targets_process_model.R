# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Production Data Science Pipelines with Targets
# Chapter     : Lab 56: TARGETS KERAS CHURN
# Module      : 02_targets_process_model.R
# Update Date : 2021/6/30
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 TARGETSによるパイプライン


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(targets)

# 仮想環境の設定
reticulate::use_condaenv("r-tf", required = TRUE)

# ディレクトリ設定
setwd("lab_56_targets_churn")


# # 1 TARGETSによるパイプライン ---------------------------------------------------

# Setup ----------------------------------------------------

# tar_script()

# Inspection ------------------------------------------------

# ＜ポイント＞
# - _targetファイルの概要を確認する


# パイプライン一覧
# --- _targetファイルに登録されている処理を一覧表示
tar_manifest()

# パイプラインの表示
tar_glimpse()

# ネットワークの表示
# --- 関数も含めて表示
tar_visnetwork()

# 古くなっている処理を確認
tar_outdated()


# Workflow ------------------------------------------------

# ＜ポイント＞
# - _targetファイルに基づいてワークフローを再定義する
#   --- 変更した場合にはこの処理が必要
#   --- _target/metaに格納されているメタデータが更新される


# 再読み込み
# --- _targetファイルを再読み込み
# --- _target/metaが更新される
tar_make()

# ネットワークの表示
# --- パイプラインをネットワーク表示
tar_visnetwork()


# Tracking Functions ----------------------------------------

tar_read(churn_file)

tar_read(churn_data)

tar_read(churn_splits)

tar_read(churn_recipe)

tar_read(run_relu)
tar_read(run_sigmoid)

tar_read(model_performance)

tar_read(best_run)

tar_read(production_model_keras)

tar_read(predictions)




