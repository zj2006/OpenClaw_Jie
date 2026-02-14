"""
常用优化测试函数
用于测试优化算法性能
"""

import numpy as np


def sphere(x):
    """
    Sphere 函数
    全局最优: f(0,...,0) = 0
    搜索空间: [-100, 100]^n
    """
    return np.sum(x**2)


def rastrigin(x):
    """
    Rastrigin 函数（多峰函数）
    全局最优: f(0,...,0) = 0
    搜索空间: [-5.12, 5.12]^n
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """
    Rosenbrock 函数（山谷函数）
    全局最优: f(1,...,1) = 0
    搜索空间: [-5, 10]^n
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley(x):
    """
    Ackley 函数（多峰函数）
    全局最优: f(0,...,0) = 0
    搜索空间: [-32.768, 32.768]^n
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank(x):
    """
    Griewank 函数（多峰函数）
    全局最优: f(0,...,0) = 0
    搜索空间: [-600, 600]^n
    """
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1


def schwefel(x):
    """
    Schwefel 函数（欺骗性函数）
    全局最优: f(420.9687,...,420.9687) ≈ 0
    搜索空间: [-500, 500]^n
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# 测试函数配置
TEST_FUNCTIONS = {
    'sphere': {
        'func': sphere,
        'bounds': (-100, 100),
        'optimal': 0,
        'description': 'Sphere 函数 - 简单凸函数'
    },
    'rastrigin': {
        'func': rastrigin,
        'bounds': (-5.12, 5.12),
        'optimal': 0,
        'description': 'Rastrigin 函数 - 多峰函数'
    },
    'rosenbrock': {
        'func': rosenbrock,
        'bounds': (-5, 10),
        'optimal': 0,
        'description': 'Rosenbrock 函数 - 山谷函数'
    },
    'ackley': {
        'func': ackley,
        'bounds': (-32.768, 32.768),
        'optimal': 0,
        'description': 'Ackley 函数 - 多峰函数'
    },
    'griewank': {
        'func': griewank,
        'bounds': (-600, 600),
        'optimal': 0,
        'description': 'Griewank 函数 - 多峰函数'
    },
    'schwefel': {
        'func': schwefel,
        'bounds': (-500, 500),
        'optimal': 0,
        'description': 'Schwefel 函数 - 欺骗性函数'
    }
}
