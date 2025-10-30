import torch
import torch.nn as nn
import multiprocessing as mp
from datetime import datetime

def gpu_compute(gpu_id, queue):
    """在指定GPU上执行密集计算"""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id} 开始计算...")
        
        # 创建大型矩阵进行计算
        size = 8192  # 矩阵大小
        
        while True:
            # 矩阵乘法 - GPU密集计算
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.matmul(a, b)
            
            # 额外的计算操作
            c = torch.nn.functional.relu(c)
            c = torch.matmul(c, c.t())
            
            # 定期输出状态
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(gpu_id) / 1024**3
                if int(datetime.now().timestamp()) % 10 == 0:
                    queue.put(f"GPU {gpu_id}: 显存使用 {memory_used:.2f} GB")
            
    except KeyboardInterrupt:
        print(f"\nGPU {gpu_id} 停止计算")
    except Exception as e:
        print(f"GPU {gpu_id} 错误: {e}")

def main():
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA支持的GPU")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    print("=" * 50)
    
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("=" * 50)
    print("开始在所有GPU上运行计算...")
    print("按 Ctrl+C 停止\n")
    
    # 创建消息队列
    queue = mp.Queue()
    
    # 为每个GPU创建进程
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_compute, args=(gpu_id, queue))
        p.start()
        processes.append(p)
    
    try:
        # 主进程监控状态
        while True:
            if not queue.empty():
                msg = queue.get()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    except KeyboardInterrupt:
        print("\n\n正在停止所有GPU计算...")
        for p in processes:
            p.terminate()
            p.join()
        print("所有GPU已停止")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()