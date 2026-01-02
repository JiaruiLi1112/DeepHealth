import argparse
import subprocess
import sys
import time
import threading
from queue import Queue

# ================= 配置区域 =================
# 全局训练参数
COMMON_ARGS = "--max_epochs 50 --pdrop 0.1"
EMBED_PATH = "icd10_sapbert_embeddings.npy"

# 定义所有实验 (与之前的 ablation 设计保持一致)
# 格式: (实验名, 独有参数字符串)
EXPERIMENTS = [
    # --- Group A: Baselines (No Pretraining) ---
    ("Exp01_Base_Exp", f"--model_type delphifork --loss_type exponential --age_encoder sinusoidal --full_cov"),
    ("Exp02_Base_Weibull",
     f"--model_type delphifork --loss_type weibull --age_encoder mlp --full_cov"),
    ("Exp03_Base_LogNormal",
     f"--model_type delphifork --loss_type lognormal --age_encoder mlp --full_cov"),

    # --- Group B: SapBERT Pretraining (Core) ---
    ("Exp04_Sap_Freeze",
     f"--model_type sapdelphi --loss_type lognormal --age_encoder mlp --full_cov --pretrained_weights_path {EMBED_PATH} --freeze_embeddings"),
    ("Exp05_Sap_Finetune",
     # 默认 finetune
     f"--model_type sapdelphi --loss_type lognormal --age_encoder mlp --full_cov --pretrained_weights_path {EMBED_PATH}"),
    ("Exp06_Sap_Weibull",
     f"--model_type sapdelphi --loss_type weibull --age_encoder mlp --full_cov --pretrained_weights_path {EMBED_PATH}"),

    # --- Group C: Data Efficiency (Lite Covariates) ---
    # 去掉了 --full_cov 即为 Lite 模式
    ("Exp07_Lite_Base", f"--model_type delphifork --loss_type lognormal --age_encoder mlp"),
    ("Exp08_Lite_Sap",
     f"--model_type sapdelphi --loss_type lognormal --age_encoder mlp --pretrained_weights_path {EMBED_PATH}"),
]
# ===========================================


def worker(gpu_id, task_queue):
    """
    工作线程：绑定一个 GPU，不断从队列取任务执行，直到队列为空。
    """
    print(f"[GPU {gpu_id}] Worker started.")

    while not task_queue.empty():
        try:
            # 非阻塞获取，防止竞争
            exp_name, exp_args = task_queue.get(block=False)
        except Exception:
            break

        print(f"🚀 [GPU {gpu_id}] Starting {exp_name}...")

        # 组装完整命令
        # 注意：这里我们通过 CUDA_VISIBLE_DEVICES 环境变量来控制通过 Python 脚本看到的 GPU
        # 这样 train.py 内部只需要使用 "cuda" 或 "cuda:0" 即可，无需修改代码
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py {COMMON_ARGS} {exp_args}"

        # 记录开始时间
        start_time = time.time()

        # 执行命令
        # capture_output=False 让日志直接打印到终端，或者你可以重定向到文件
        try:
            # 建议将日志重定向到文件，避免终端混乱
            with open(f"logs/{exp_name}.log", "w") as log_file:
                subprocess.run(cmd, shell=True, check=True,
                               stdout=log_file, stderr=subprocess.STDOUT)
            status = "✅ Done"
        except subprocess.CalledProcessError:
            status = "❌ Failed"

        duration = time.time() - start_time
        print(
            f"{status} [GPU {gpu_id}] {exp_name} finished in {duration:.1f}s. Check logs/{exp_name}.log")

        task_queue.task_done()

    print(f"[GPU {gpu_id}] No more tasks. Worker exiting.")


def main():
    parser = argparse.ArgumentParser(description="Parallel Experiment Runner")
    parser.add_argument("--gpus", nargs="+", type=int, required=True,
                        help="List of GPU IDs to use, e.g., --gpus 0 1 2 3")
    args = parser.parse_args()

    # 1. 准备日志目录
    import os
    os.makedirs("logs", exist_ok=True)

    # 2. 创建任务队列
    task_queue = Queue()
    for exp in EXPERIMENTS:
        task_queue.put(exp)

    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Available GPUs: {args.gpus}")
    print("logs will be saved to ./logs/ directory.")
    print("-" * 40)

    # 3. 为每个 GPU 启动一个线程
    threads = []
    for gpu_id in args.gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)
        # 稍微错开启动时间，避免瞬间 IO 峰值
        time.sleep(2)

    # 4. 等待所有线程结束
    for t in threads:
        t.join()

    print("-" * 40)
    print("🎉 All experiments finished!")


if __name__ == "__main__":
    main()
