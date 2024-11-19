@echo off
REM ============================================
REM           运行主脚本: run_attack.bat
REM ============================================

REM 设置变量
set "dataset=pubmed"
set "data_folder=data\Planetoid\%dataset%"
set "base_mod=Deep-featpre"
set "down_task=node_classification"
set "budget=20"

set "transf_mod=Deep-featpre"
set "transf_task=node_classification"
set "preT_mod=Deep-gin"
set "saved_model=attack_models\%down_task%\%dataset%\model-%base_mod%"
set "saved_emb_model=attack_models\%transf_task%\%dataset%\model-%transf_mod%"

set "directed=0"
set "lr=0.01"
set "batch_size=2" REM 10
set "reward_type=emb_silh"
set "reward_state=marginal"
REM gm=mean_field
set "num_hops=2"
set "mu_type=e2e_embeds"
REM location for pre-trained embeddings
set "embeds=attack_models\emb_models\%dataset%\model-%preT_mod%.npy"
set "dqn_hidden=16"
set "embed_dim=16"
set "discount=0.9"
set "q_nstep=2"
set "num_epds=5" REM 500
set "sample_size=2" REM 10

set "output_base=results\target_nodes\%dataset%-%directed%-%transf_mod%-%transf_task%"

REM 根据 mu_type 设置 save_fold
if /I "%mu_type%"=="preT_embeds" (
    set "save_fold=rl-%lr%-%discount%_mu-preT-%preT_mod%_q-%q_nstep%-%dqn_hidden%"
) else (
    set "save_fold=rl-%lr%-%discount%_mu-%mu_type%_q-%q_nstep%-%dqn_hidden%"
)
set "output_root=%output_base%\%save_fold%"

REM 创建输出目录（如果不存在）
if not exist "%output_root%" (
    mkdir "%output_root%"
)

REM 显示设置的变量（可选，便于调试）
echo Data Folder: %data_folder%
echo Dataset: %dataset%
echo Base Model: %base_mod%
echo Down Task: %down_task%
echo Budget: %budget%
echo Transfer Model: %transf_mod%
echo Transfer Task: %transf_task%
echo Pre-trained Model: %preT_mod%
echo Saved Model Path: %saved_model%
echo Saved Embedding Model Path: %saved_emb_model%
echo Directed: %directed%
echo Learning Rate: %lr%
echo Batch Size: %batch_size%
echo Reward Type: %reward_type%
echo Reward State: %reward_state%
echo Number of Hops: %num_hops%
echo Mu Type: %mu_type%
echo Embeds Path: %embeds%
echo DQN Hidden: %dqn_hidden%
echo Embed Dimension: %embed_dim%
echo Discount: %discount%
echo Q N-Step: %q_nstep%
echo Number of Epochs: %num_epds%
echo Sample Size: %sample_size%
echo Output Root: %output_root%

REM 运行 Python 脚本
python main.py ^
    -directed %directed% ^
    -budget %budget% ^
    -data_folder %data_folder% ^
    -dataset %dataset% ^
    -down_task %down_task% ^
    -saved_model %saved_model% ^
    -saved_emb_model %saved_emb_model% ^
    -save_dir %output_root% ^
    -embeds %embeds% ^
    -learning_rate %lr% ^
    -num_hops %num_hops% ^
    -mu_type %mu_type% ^
    -embed_dim %embed_dim% ^
    -dqn_hidden %dqn_hidden% ^
    -reward_type %reward_type% ^
    -reward_state %reward_state% ^
    -batch_size %batch_size% ^
    -num_epds %num_epds% ^
    -sample_size %sample_size% ^
    -q_nstep %q_nstep% ^
    -discount %discount% ^
    -phase test ^
    -seed 123 ^
    -nprocs 10 ^
    -lcc %*  REM 将 -lcc 和所有额外参数放在同一行

REM 如果需要，可以取消下面四行的注释以启用其他选项
REM -save_sols_only
REM -save_sols_file sols_gcn
REM -target_perc 0.1
REM -device cuda

REM 结束脚本
echo.
echo 脚本运行完成。
pause
