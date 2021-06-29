# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Production Data Science Pipelines with Targets
# Chapter     : Lab 56: TARGETS KERAS CHURN
# Module      : 01_keras_installation.R
# Update Date : 2021/6/28
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************

# ライブラリ
library(reticulate)


# PYTHON KERAS SETUP ----------------------------------------------------

# インストール
reticulate::conda_install(
    envname = "r-tf",
    packages = c("tensorflow", "keras", "h5py", "pyyaml")
)

## Select / Check python interpreter ----
#  - Tools > Project Options > Python > Select Interpreter ("r-tf")

# 仮想環境を選択
reticulate::use_condaenv("r-tf", required = TRUE)

# コンフィグ確認
# --- 指定した仮想環境が選択されていることを確認
reticulate::py_config()


## Conda Environment YAML ----
#   - Open environment_r-tf.yml file to see specific python package versions


