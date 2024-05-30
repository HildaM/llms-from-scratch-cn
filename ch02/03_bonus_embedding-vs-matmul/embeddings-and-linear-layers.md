# Embedding层和 Linear层

- 在PyTorch中，嵌入层（Embedding layers）实现了执行矩阵乘法的线性层的相同功能；我们使用嵌入层的原因是为了提高计算效率。
- 我们将逐步使用PyTorch中的代码示例来查看这种关系。


```python
import torch
print("PyTorch version:", torch.__version__)
```

    PyTorch version: 2.2.0+cu121
    

## Using nn.Embedding


```python
# 假设我们有以下 3 个训练样本，
# 这些样本可能表示语言模型（LM）上下文中的标记ID
idx = torch.tensor([2, 3, 1])

# 嵌入矩阵的行数可以通过获取最大标记ID + 1 来确定。
# 如果最高的标记ID是3，则我们希望有4行，对应可能的
# 标记ID 0, 1, 2, 3
num_idx = max(idx) + 1

# 所需的嵌入维度是一个超参数
out_dim = 5
```

- 实现一个简单的嵌入层


```python
# 为了可重复性，我们使用随机种子，
# 因为嵌入层的权重是用小的随机值初始化的
torch.manual_seed(123)

# 创建一个嵌入层，指定输入维度为 num_idx，输出维度为 out_dim
embedding = torch.nn.Embedding(num_idx, out_dim)
```

查看嵌入权重数据情况


```python
embedding.weight
```




    Parameter containing:
    tensor([[ 0.3374, -0.1778, -0.3035, -0.5880,  1.5810],
            [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015],
            [ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953]], requires_grad=True)



- 使用嵌入层来获取具有ID 1的训练样本的向量表示


```python
embedding(torch.tensor([1]))
```




    tensor([[ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
           grad_fn=<EmbeddingBackward0>)



- 下面是底层操作的可视化

<img src="images/1.png" width="400px">

- 同样，我们可以使用嵌入层来获取具有ID 2的训练样本的向量表示：


```python
embedding(torch.tensor([2]))
```




    tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315]],
           grad_fn=<EmbeddingBackward0>)



<img src="images/2.png" width="400px">

- 现在，让我们将之前定义的所有训练样本转换：


```python
# 将原先的第三行变成现在的第一行，第四行变成现在的第二行，第二行变成现在的第三行
idx = torch.tensor([2, 3, 1])
embedding(idx)
```




    tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
            [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
           grad_fn=<EmbeddingBackward0>)



- Under the hood, it's still the same look-up concept:

<img src="images/3.png" width="450px">

## 使用 nn.Linear

- 接下来，我们将使用One-Hot编码，与embedding 层一样，在 `nn.Linear` 层进行操作
- 首先，我们将标记ID转换为One-Hot表示：


```python
onehot = torch.nn.functional.one_hot(idx)
onehot
```




    tensor([[0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]])



- 接下来，我们使用矩阵乘法$X W^\top$ 来初始化一个Linear层


```python
torch.manual_seed(123)
# 初始化一个Linear层，该层的权重矩阵是由 num_idx（输入维度）到 out_dim（输出维度）的一个线性层，而且没有偏置项
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print(linear.weight)
```

    Parameter containing:
    tensor([[-0.2039,  0.0166, -0.2483,  0.1886],
            [-0.4260,  0.3665, -0.3634, -0.3975],
            [-0.3159,  0.2264, -0.1847,  0.1871],
            [-0.4244, -0.3034, -0.1836, -0.0983],
            [-0.3814,  0.3274, -0.1179,  0.1605]], requires_grad=True)
    

- 请注意，PyTorch中的`linear`层也是用小的随机权重进行初始化的。为了与上面的 `Embedding` 层进行直接比较，我们必须使用相同的小随机权重，这就是我们在这里重新分配它们的原因：


```python
# linear 层的权重就被重新赋值为与 embedding 层相同的小随机权重，以确保它们具有相同的初始化。这是为了使它们在后续操作中可以进行直接比较。
linear.weight = torch.nn.Parameter(embedding.weight.T.detach())
```

- 现在，我们可以使用线性层处理输入的One-Hot编码表示：


```python
linear(onehot.float())
```




    tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
            [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]], grad_fn=<MmBackward0>)



正如我们所看到的，这与我们使用嵌入层时得到的结果完全相同：


```python
embedding(idx)
```




    tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
            [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
           grad_fn=<EmbeddingBackward0>)



- 底层发生的计算如下，针对第一个训练样本的标记ID：

<img src="images/4.png" width="450px">

- 以及对于第二个训练样本的标记ID：

<img src="images/5.png" width="450px">

- 
由于每个独热编码行中除了一个索引外都为0（设计如此），这个矩阵乘法本质上就是对独热编码元素的查找
- 。在独热编码上使用矩阵乘法与使用嵌入层查找是等效的，但如果我们使用大型嵌入矩阵，这种方法可能效率较低，因为有很多不必要的零乘法。


```python

```
