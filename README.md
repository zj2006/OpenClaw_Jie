# PSO Algorithm Test

粒子群优化算法 (Particle Swarm Optimization) 的 Python 实现

## 简介

粒子群优化算法是一种基于群体智能的优化算法，模拟鸟群觅食行为。本项目提供了完整的 PSO 实现，包括多种测试函数和性能评估工具。

## 特性

- ✅ 标准 PSO 实现
- ✅ 自适应惯性权重
- ✅ 6 种经典优化测试函数（Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel）
- ✅ 性能基准测试工具
- ✅ 可视化收敛曲线和对比分析
- ✅ 支持自定义目标函数

## 文件结构

```
pso-algorithm-test/
├── pso.py                    # PSO 核心算法
├── benchmark_functions.py    # 测试函数库
├── benchmark.py              # 性能基准测试
├── README.md                 # 说明文档
└── requirements.txt          # 依赖包
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基础使用

```bash
python pso.py
```

### 2. 性能基准测试

```bash
python benchmark.py
```

这将在 6 种测试函数上运行 PSO，并生成对比图。

### 3. 自定义使用

```python
from pso import PSO
from benchmark_functions import rastrigin

# 创建 PSO 实例
pso = PSO(
    n_particles=30,      # 粒子数量
    n_dimensions=10,     # 问题维度
    max_iter=200,        # 最大迭代次数
    bounds=(-5.12, 5.12),# 搜索空间
    adaptive_w=True      # 使用自适应惯性权重
)

# 执行优化
best_pos, best_score, history = pso.optimize(objective_func=rastrigin)

print(f"最优位置: {best_pos}")
print(f"最优值: {best_score}")

# 绘制收敛曲线
pso.plot_convergence()
```

## 参数说明

### PSO 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_particles` | 粒子数量 | 30 |
| `n_dimensions` | 问题维度 | 2 |
| `max_iter` | 最大迭代次数 | 100 |
| `w` | 惯性权重（固定模式） | 0.7 |
| `c1` | 个体学习因子 | 1.5 |
| `c2` | 社会学习因子 | 1.5 |
| `bounds` | 搜索空间边界 | (-10, 10) |
| `adaptive_w` | 自适应惯性权重 | True |
| `w_min` | 最小惯性权重 | 0.4 |
| `w_max` | 最大惯性权重 | 0.9 |

### 测试函数

| 函数 | 全局最优 | 搜索空间 | 特点 |
|------|----------|----------|------|
| Sphere | f(0,...,0) = 0 | [-100, 100]^n | 简单凸函数 |
| Rastrigin | f(0,...,0) = 0 | [-5.12, 5.12]^n | 多峰函数 |
| Rosenbrock | f(1,...,1) = 0 | [-5, 10]^n | 山谷函数 |
| Ackley | f(0,...,0) = 0 | [-32.768, 32.768]^n | 多峰函数 |
| Griewank | f(0,...,0) = 0 | [-600, 600]^n | 多峰函数 |
| Schwefel | f(420.9687,...) ≈ 0 | [-500, 500]^n | 欺骗性函数 |

## 算法说明

### 标准 PSO

粒子位置和速度更新公式：

```
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

其中：
- `v`: 粒子速度
- `x`: 粒子位置
- `w`: 惯性权重
- `c1, c2`: 学习因子
- `r1, r2`: [0,1] 随机数
- `pbest`: 个体最优位置
- `gbest`: 全局最优位置

### 自适应惯性权重

线性递减策略：

```
w(t) = w_max - (w_max - w_min) · t / T
```

其中 `T` 是最大迭代次数。这种策略在初期鼓励全局搜索，后期加强局部搜索。

## 输出示例

```
==================================================
粒子群优化算法 (PSO) 演示
==================================================
迭代 0: 最优值 = 45.234567
迭代 10: 最优值 = 12.345678
...
迭代 90: 最优值 = 0.000123

==================================================
优化结果:
最优位置: [0.0001 -0.0002]
最优值: 0.0000000123
==================================================
```

## License

MIT

## 作者

zj2006

## 参考文献

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
2. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.

