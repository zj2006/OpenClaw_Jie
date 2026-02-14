"""
粒子群优化算法 (Particle Swarm Optimization, PSO)
用于求解优化问题的简单实现
"""

import numpy as np
import matplotlib.pyplot as plt


class PSO:
    """粒子群优化算法类"""
    
    def __init__(self, n_particles=30, n_dimensions=2, max_iter=100, 
                 w=0.7, c1=1.5, c2=1.5, bounds=(-10, 10)):
        """
        初始化 PSO 参数
        
        Args:
            n_particles: 粒子数量
            n_dimensions: 问题维度
            max_iter: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            bounds: 搜索空间边界
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        
        # 初始化粒子位置和速度
        self.positions = np.random.uniform(
            bounds[0], bounds[1], 
            (n_particles, n_dimensions)
        )
        self.velocities = np.random.uniform(
            -abs(bounds[1] - bounds[0]) * 0.1, 
            abs(bounds[1] - bounds[0]) * 0.1,
            (n_particles, n_dimensions)
        )
        
        # 个体最优和全局最优
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(n_particles, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        
        # 记录历史
        self.history = []
    
    def objective_function(self, x):
        """
        目标函数：Sphere函数（测试用）
        f(x) = sum(x_i^2)
        全局最优解在 x = [0, 0, ..., 0]，f(x) = 0
        """
        return np.sum(x**2)
    
    def optimize(self, objective_func=None):
        """
        执行优化过程
        
        Args:
            objective_func: 自定义目标函数（可选）
        
        Returns:
            最优位置, 最优值, 历史记录
        """
        if objective_func is None:
            objective_func = self.objective_function
        
        for iteration in range(self.max_iter):
            # 评估所有粒子
            for i in range(self.n_particles):
                score = objective_func(self.positions[i])
                
                # 更新个体最优
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()
            
            # 更新速度和位置
            r1 = np.random.random((self.n_particles, self.n_dimensions))
            r2 = np.random.random((self.n_particles, self.n_dimensions))
            
            self.velocities = (
                self.w * self.velocities +
                self.c1 * r1 * (self.pbest_positions - self.positions) +
                self.c2 * r2 * (self.gbest_position - self.positions)
            )
            
            self.positions = self.positions + self.velocities
            
            # 边界处理
            self.positions = np.clip(
                self.positions, 
                self.bounds[0], 
                self.bounds[1]
            )
            
            # 记录历史
            self.history.append(self.gbest_score)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 最优值 = {self.gbest_score:.6f}")
        
        return self.gbest_position, self.gbest_score, self.history
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.xlabel('迭代次数')
        plt.ylabel('最优值')
        plt.title('PSO 收敛曲线')
        plt.grid(True)
        plt.savefig('convergence.png')
        print("收敛曲线已保存到 convergence.png")


def main():
    """主函数：演示 PSO 算法"""
    print("=" * 50)
    print("粒子群优化算法 (PSO) 演示")
    print("=" * 50)
    
    # 创建 PSO 实例
    pso = PSO(
        n_particles=30,
        n_dimensions=2,
        max_iter=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        bounds=(-10, 10)
    )
    
    # 执行优化
    best_pos, best_score, history = pso.optimize()
    
    print("\n" + "=" * 50)
    print("优化结果:")
    print(f"最优位置: {best_pos}")
    print(f"最优值: {best_score:.10f}")
    print("=" * 50)
    
    # 绘制收敛曲线
    pso.plot_convergence()


if __name__ == "__main__":
    main()
