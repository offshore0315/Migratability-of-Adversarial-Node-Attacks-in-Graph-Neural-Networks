@echo off
REM 设置变量
set "dataset=pubmed"
set "model_name=Deep"
set "layer=gcn"
set "hidden_dim=64"
set "down_task=node_classification"
set "model_save_name=model-%model_name%-featpre"

set "directed=0"
set "dropout_rate=0.5"
set "n_epochs=20"
set "lr=0.01"
set "batch_size=100"
set "patience=60"
set "save_embs=0"

REM 切换到 attack_models 目录
cd attack_models

REM 设置并创建 down_task\dataset 目录
set "output_root=%down_task%\%dataset%"
if not exist "%output_root%" (
    mkdir "%output_root%"
)

REM 设置并创建 emb_models\dataset 目录
set "output_root=emb_models\%dataset%"
if not exist "%output_root%" (
    mkdir "%output_root%"
)

REM 运行 Python 脚本
python train_model.py ^
    -dataset %dataset% ^
    -model_name %model_name% ^
    -layer %layer% ^
    -down_task %down_task% ^
    -model_save_name %model_save_name% ^
    -dropout_rate %dropout_rate% ^
    -n_epochs %n_epochs% ^
    -hidden_layers %hidden_dim% %hidden_dim% ^
    -directed %directed% ^
    -lr %lr% ^
    -patience %patience% ^
    -save_embs %save_embs% ^
    -batch_size %batch_size% ^
    -device cpu ^  REM 修改这里为 CPU
    -lcc ^
    %*

REM 如果需要删除临时文件，可以取消下面一行的注释
REM del temp*.pt
