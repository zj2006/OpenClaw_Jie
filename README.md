# PSO Algorithm Test

粒子群优化算法 (Particle Swarm Optimization) 的 Python 实现

## 简介

粒子群优化算法是一种基于群体智能的优化算法，模拟鸟群觅食行为。

## 特性

- ✅ 简洁的 PSO 实现
- ✅ 支持自定义目标函数
- ✅ 可视化收敛曲线
- ✅ 易于扩展和修改

## 安装

```bash
pip install numpy matplotlib
```

## 使用方法

```bash
python pso.py
```

## 参数说明

- `n_particles`: 粒子数量 (默认: 30)
- `n_dimensions`: 问题维度 (默认: 2)
- `max_iter`: 最大迭代次数 (默认: 100)
- `w`: 惯性权重 (默认: 0.7)
- `c1`: 个体学习因子 (默认: 1.5)
- `c2`: 社会学习因子 (默认: 1.5)

## 示例

```python
from pso import PSO

# 创建 PSO 实例
pso = PSO(n_particles=30, n_dimensions=2, max_iter=100)

# 执行优化
best_pos, best_score, history = pso.optimize()

print(f"最优位置: {best_pos}")
print(f"最优值: {best_score}")
```

## License

MIT
