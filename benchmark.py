"""
PSO 算法性能对比测试
在多个测试函数上评估 PSO 性能
"""

import numpy as np
import matplotlib.pyplot as plt
from pso import PSO
from benchmark_functions import TEST_FUNCTIONS


def run_benchmark(func_name, n_runs=10, n_dimensions=10):
    """
    在指定测试函数上运行基准测试
    
    Args:
        func_name: 测试函数名称
        n_runs: 运行次数
        n_dimensions: 问题维度
    
    Returns:
        结果字典
    """
    config = TEST_FUNCTIONS[func_name]
    func = config['func']
    bounds = config['bounds']
    optimal = config['optimal']
    
    results = []
    best_scores = []
    
    print(f"\n{'='*60}")
    print(f"测试函数: {func_name} - {config['description']}")
    print(f"维度: {n_dimensions}, 运行次数: {n_runs}")
    print(f"{'='*60}")
    
    for run in range(n_runs):
        pso = PSO(
            n_particles=30,
            n_dimensions=n_dimensions,
            max_iter=200,
            bounds=bounds,
            adaptive_w=True
        )
        
        best_pos, best_score, history = pso.optimize(objective_func=func)
        results.append(history)
        best_scores.append(best_score)
        
        error = abs(best_score - optimal)
        print(f"运行 {run+1:2d}: 最优值 = {best_score:.6e}, 误差 = {error:.6e}")
    
    # 统计结果
    best_scores = np.array(best_scores)
    mean_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    min_score = np.min(best_scores)
    
    print(f"\n统计结果:")
    print(f"  平均值: {mean_score:.6e}")
    print(f"  标准差: {std_score:.6e}")
    print(f"  最优值: {min_score:.6e}")
    print(f"  理论最优: {optimal:.6e}")
    
    return {
        'func_name': func_name,
        'results': results,
        'best_scores': best_scores,
        'mean': mean_score,
        'std': std_score,
        'min': min_score,
        'optimal': optimal
    }


def plot_comparison(benchmark_results):
    """绘制多个测试函数的对比图"""
    n_funcs = len(benchmark_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(benchmark_results):
        ax = axes[idx]
        func_name = result['func_name']
        results = result['results']
        
        # 绘制所有运行的收敛曲线
        for history in results:
            ax.plot(history, alpha=0.3, color='blue')
        
        # 绘制平均收敛曲线
        mean_history = np.mean(results, axis=0)
        ax.plot(mean_history, color='red', linewidth=2, label='平均')
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最优值')
        ax.set_title(f'{func_name}\n平均: {result["mean"]:.2e} ± {result["std"]:.2e}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150)
    print(f"\n对比图已保存到 benchmark_comparison.png")


def main():
    """主函数"""
    print("=" * 60)
    print("PSO 算法性能基准测试")
    print("=" * 60)
    
    # 选择要测试的函数
    test_functions = ['sphere', 'rastrigin', 'rosenbrock', 
                     'ackley', 'griewank', 'schwefel']
    
    benchmark_results = []
    
    for func_name in test_functions:
        result = run_benchmark(func_name, n_runs=10, n_dimensions=10)
        benchmark_results.append(result)
    
    # 绘制对比图
    plot_comparison(benchmark_results)
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
