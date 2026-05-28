# 感知机

> 感知机是神经网络的原子。把它拆开，你会看到权重、偏置，以及一次决策。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 学习目标

- 从零开始用 Python 实现感知机，包括权重更新规则和阶跃激活函数
- 解释为什么单个感知机只能解决线性可分问题，并演示 XOR 失败案例
- 通过组合 OR、NAND 和 AND 门构造多层感知机来解决 XOR
- 使用 sigmoid 激活和反向传播训练一个双层网络，让它自动学会 XOR

## 问题

你已经知道向量和点积。你也知道矩阵会把输入变成输出。但机器到底是怎么*学会*该用哪种变换的？

感知机给出了答案。它是最简单的学习机器：取一些输入，与权重相乘，加上偏置，然后做一个二元决策。再调整。就这么简单。所有出现过的神经网络，都是把这个想法一层层堆叠起来。

理解感知机，就等于理解代码里的“学习”到底是什么意思：不断调整数字，直到输出与现实匹配。

## 概念

### 一个神经元，一次决策

感知机接收 n 个输入，把每个输入乘上对应权重，再求和，加上偏置，然后把结果送入激活函数。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

阶跃函数很“粗暴”：如果加权和再加偏置后的结果 >= 0，就输出 1；否则输出 0。

```text
step(z) = 1  if z >= 0
           0  if z < 0
```

这是一个线性分类器。权重和偏置定义了一条直线（更高维时是一个超平面），把输入空间切分成两个区域。

### 决策边界

对于两个输入，感知机会在二维空间里画出一条线：

```text
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

线一侧的所有点输出 0，另一侧的所有点输出 1。训练的过程，就是不断移动这条线，直到它能正确分开各个类别。

### 学习规则

感知机的学习规则很简单：

```text
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

如果预测正确，`error = 0`，参数不变。如果它预测成 0 但正确答案应该是 1，权重就会增大。如果它预测成 1 但正确答案应该是 0，权重就会减小。学习率控制每次调整的幅度。

### XOR 问题

问题也正是在这里暴露出来。看这些逻辑门：

```text
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND 和 OR 是线性可分的：你可以画一条线，把 0 和 1 分开。XOR 不行。没有任何一条单独的直线，能够把 `[0,1]` 和 `[1,0]` 与 `[0,0]` 和 `[1,1]` 分开。

```text
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是一个根本性的限制。单个感知机只能解决线性可分问题。Minsky 和 Papert 在 1969 年证明了这一点，这个结论几乎让神经网络研究沉寂了整整十年。

修复方法是：把感知机堆成多层。多层感知机可以通过组合两个线性决策，形成一个非线性决策，从而解决 XOR。

## 动手实现

### 第 1 步：`Perceptron` 类

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### 第 2 步：在逻辑门数据上训练

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### 第 3 步：观察 XOR 如何失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

它永远不会收敛。这就是一个硬证据：单个感知机不可能学会 XOR。

### 第 4 步：用两层解决 XOR

诀窍是：`XOR = (x1 OR x2) AND NOT (x1 AND x2)`。组合三个感知机：

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

四种情况全部正确。把感知机堆成多层之后，就能形成单个感知机无法表达的决策边界。

### 第 5 步：训练一个双层网络

第 4 步里我们是手工指定权重的。对于 XOR 这没问题，但在真实问题中，你事先并不知道正确权重是什么。解决办法是：把阶跃函数换成 sigmoid，并通过反向传播自动学习权重。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

和第 4 步相比，这里有两个关键区别。第一，sigmoid 替换了阶跃函数，它是平滑的，因此梯度存在。第二，`train` 方法会把误差从输出层反向传播到隐藏层，并按照每个权重对误差的贡献比例来调整它。这就是 20 行代码版的反向传播。

这也是通向第 03 课的桥梁。`d_output` 和 `hidden_deltas` 背后的数学，本质上是把链式法则应用到网络图上。我们会在那里正式推导。

## 用起来

你刚刚从零实现的所有东西，在实际库里只需要一次导入：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你写的 30 行 `Perceptron` 类做的是同一件事。`sklearn` 版本额外加入了收敛检查、多种损失函数以及对稀疏输入的支持，但核心循环完全一样：加权求和、阶跃函数、在出错时更新权重。

真正的差异出现在规模扩大之后。生产级网络中会发生这些变化：

- 阶跃函数会变成 sigmoid、ReLU 或其他平滑激活函数
- 权重通过反向传播自动学习（第 03 课）
- 网络层数会更深：3 层、10 层、100+ 层
- 核心原理不变：每一层都会从前一层的输出中构造新的特征

单个感知机只能画直线。把它们堆起来，你就能画出任意形状。

## 交付物

本课会产出：
- `outputs/skill-perceptron.md` - 一个说明何时需要单层架构、何时需要多层架构的技能文档

## 练习

1. 在 NAND 门数据上训练一个感知机（通用门，任何逻辑电路都能由 NAND 构成）。验证它的权重和偏置是否形成了有效的决策边界。
2. 修改 `Perceptron` 类，让它在每个 epoch 记录决策边界（`w1*x1 + w2*x2 + b = 0`）。打印这条线在 AND 门训练过程中的移动方式。
3. 构建一个 3 输入感知机：只有当 3 个输入中至少有 2 个为 1 时才输出 1（多数投票函数）。它是线性可分的吗？为什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Perceptron | "A fake neuron" | 一个线性分类器：输入与权重做点积，再加偏置，最后通过阶跃函数 |
| Weight | "How important an input is" | 一个乘数，用来缩放每个输入对最终决策的贡献 |
| Bias | "The threshold" | 一个常数，用于平移决策边界，使感知机即使在零输入时也可能激活 |
| Activation function | "The thing that squishes values" | 加权求和后应用的函数：感知机里是阶跃函数，现代网络里通常是 sigmoid/ReLU |
| Linearly separable | "You can draw a line between them" | 指一个数据集可以被单个超平面完美分开各个类别 |
| XOR problem | "The thing perceptrons can't do" | 证明单层网络无法学习非线性可分函数的经典案例 |
| Decision boundary | "Where the classifier switches" | 超平面 `w*x + b = 0`，它把输入空间分成两类 |
| Multi-layer perceptron | "A real neural network" | 按层堆叠的感知机，每一层的输出作为下一层的输入 |

## 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) -- 开创这一切的原始论文
- Minsky & Papert, "Perceptrons" (1969) -- 证明单层网络无法解决 XOR，并让感知机研究沉寂十年的著作
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- 免费在线，最适合理解感知机如何组合成网络的可视化讲解
