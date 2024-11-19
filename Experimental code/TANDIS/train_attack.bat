@echo off
REM 设置变量
set "dataset=pubmed"
set "data_folder=data/Planetoid/%dataset%"
set "layer=gcn"

set "transf_mod=Deep-featpre"
set "transf_task=node_classification" REM #link_prediction
set "saved_emb_model=attack_models\%transf_task%\%dataset%\model-%transf_mod%"

set "directed=0"
set "lr=0.01"
set "batch_size=2" REM 10
set "reward_type=emb_silh"
set "reward_state=marginal"
REM gm=mean_field
set "num_hops=2"
set "mu_type=e2e_embeds"
set "preT_mod=Deep-gcn"
REM location for pre-trained embeddings
set "embeds=attack_models\emb_models\%dataset%\model-%preT_mod%.npy"
set "embed_dim=16"
set "dqn_hidden=16"
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

REM 运行 Python 脚本
python main.py ^
    -directed %directed% ^
    -data_folder %data_folder% ^
    -dataset %dataset% ^
    -down_task %transf_task% ^
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
    -phase train ^
    -seed 123 ^
    -nprocs 10 ^
    -lcc ^
    %*

REM 如果需要，可以取消下面两行的注释以启用 GPU 上下文或其他选项
REM -ctx gpu
REM -nprocs 10
