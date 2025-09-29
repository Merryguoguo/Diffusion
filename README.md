# Diffusion
Basic, Pop Diffusion Tech

### :star: Tech Details
#### :droplet: 将时间嵌入（time emdedding）输入UNet后，UNet是如何调整其对于不同时间步、不同噪声强度的响应的？
> 对于Diffusion Models来说，同一个输入图像（或潜在变量）在不同的时间步t看上去是完全不同的（比如，t=1时噪声少，t=T时噪声多），
> 模型必须知道当前是出于哪个时间步，才能正确地去噪。

1. **Time Embedding**
   > 将离散的时间步t(1~1000) :arrow_right: 某种方式映射 :arrow_right: 高维稠密的向量表示，\
   > 类似于词嵌入（word embedding）。\
   > 功能：是模型能够理解不同时间步的语义（即不同的噪声强度和去噪目标）
   
   :unlock: 常见嵌入方法：\
   a. 正弦 / 余弦位置编码（类似 Transformer）
      > 把时间步当作“位置”pos，生成一组正弦/余弦频率特征
      > PE(pos, 2i) = sin(pos/10000^(2i/d)),
      > PE(pos, 2i+!) = cos(pos/10000^(2i/d))
      
   b. 可学习的线性投影（MLP Embedding）
      > 将t（标量或one-hot）通过一个或多个全连接层映射为高维向量
      > 更灵活，训练过程中可调整
      
   c. 组合方式
      > 通过正弦编码，再通过可学习MLP进一步映射

   :wrench: 在实际实现中（比如 Stable Diffusion、DDPM、DDIM），  
   时间嵌入通常是一个可训练的、密集的向量表示，
   维度与模型中间层的通道数相匹配，以便与图像特征融合。
   
2. **Time Embedding如何注入UNet？如果调整其对不同时间步的响应？**  
   :dart: **核心思想**
   > UNet在处理某一时刻的噪声数据时，  
   > 不仅要看图像本身的内容，还要知道“现在处于扩散过程的哪个时间点t”，   
   > 从而调整其内部特征表示，以适应不同噪声强度的输入。
   
   **Step 1: 时间嵌入生成（基于时间步t）**  
   > 在推理或训练时，对于每一个时间步t，首先计算或查表得到一个时间嵌入向量（time embedding vector），  
   > 通常维度为[1, embed_dim]或扩展为[1, embed_dim, 1, 1]以匹配空间维度。

   **Step2: 时间嵌入被注入到UNet的多个层次中**  
   :stars: 注入到UNet的每个时空卷积块（如ResNet Block/Downsample/Upsample Block）
   > 在每个ResBlock（或类似的卷积块）中，除了输入图像特征外，还会将时间嵌入作为额外的条件信息输入  
   > 具体方式：  
   > a. 通过Time Embedding与图像特征拼接（concat）后送入卷积  
   > b. 通过Time Emdedding先经过一个MLP或Transformer Layer生成一组“时间相关的权重/偏置”，再对图像特征进行调整（如AdaGN, FiLM, TimeMix等机制）  
   > c. 更常见的是：时间嵌入被送入一组“时间嵌入MLP”或“时间卷积”，生成一组与图像通道对齐的调制参数，然后对主路径的特征进行调制（类似条件批归一化/特征变换）  
   
   :stars: 典型实现：通过Time Emdedding调整 BatchNorm / GroupNorm / AdaGN 等
   > 比如在某些视线中，时间嵌入会用来生成**条件伽马（scale）和beta（shift）参数**，  
   > 用于对特征图进行**仿射变换**，  
   > 从而调整每一层对不同时间步的响应。

   :stars: 通过 Cross Attention（如果使用Transformer结构）
   > 如果你的UNet中含有Transformer Block或Cross Attention层（比如在Stable Diffusion中），  
   > 时间嵌入可以作为额外的Key/Value或条件Token，控制注意力机制对不同时间信息的关注。

   **Step 3: 时间嵌入让模型感知“当前噪声水平”，从而调整去噪策略**
   > 当t较小（如t=5）时，图像中的噪声较少，模型需要做的是：细微调整，保留细节  
   > 当t较大（如t=999）时，图像基本上时纯噪声，模型需要做的是：大胆预测结构、生成整体布局  
   
   > 通过时间嵌入，UNet能够：  
   > a. 调整其特征提取的强度  
   > b. 改变去噪的偏置（比如更倾向于生成结构还是纹理）  
   > c. 控制不同层次对噪声的敏感度  

   :page_with_curl: 总结：时间嵌入让同一个输入图像，在不同时间步t下，经过UNet时能得到不同的特征响应，  
   从而使模型学会针对不同噪声强度进行恰当的去噪。  
   
3. **总结：时间嵌入如何调整UNet对不同噪声强度的响应？**
   > 时间嵌入将时间步t编码为高维向量，  
   > 并注入到UNet的多个层次中（如ResBlock、Normalization层、Attention层等）  
   > 使模型能够感知当前所处的扩散时间步，从而动态调整其特征表示与去噪策略，    
   > 以适应不同时间步下不同的噪声强度和去噪目标。

   <img src="https://github.com/Merryguoguo/Diffusion/blob/main/Diffusion1.PNG" width="500px">


#### :droplet: 如何在UNet中实现FiLM(Feature-wise Linear Modulation)或AdaGN(Adaptive Group Normalization)等时间条件调制机制？
> 使模型能够根据时间步（或其他条件）动态调整特征表示  

:bulb: **条件生成模型** （Conditional Generation）中如何将**额外信息**（如时间步t、文本嵌入、类别标签等）注入到神经网络中，从而控制其输出行为，这在扩散模型（如**Stable Diffusion**）、风格迁移、**可控生成任务**中都有广泛应用。  

> 在扩散模型中，我们希望UNet不仅能处理**图像数据**，  
> 还能根据当前的**时间步t（或其他条件，如文本、类别）** 动态调整其**内部特征表示**，以适应**不同噪声强度**或**生成目标**。

:green_apple: **条件调制机制**  
> 将时间嵌入等信息以某种方式融合进UNet的**特征图**中，  
> 控制特征的缩放（scale）和偏移（shift）(即**特征变换**),  
> 或者控制**归一化**的行为。  

<img src="https://github.com/Merryguoguo/Diffusion/blob/main/Diffusion_%E6%9D%A1%E4%BB%B6%E6%8E%A7%E5%88%B6%E6%9C%BA%E5%88%B6.png" width='500px'>

:white_check_mark: 方法一：FilM（Feature-wise Linear Modulation）——**逐通道**线性调试  
:dart: **核心思想**  
> 对于每一个特征图通道，  
> 使用条件生成一对参数（ γ 和 β ），   
> 对该通道的特征进行线性变换：输出 = γ × 特征 + β。  

相当于让**条件信息**控制**该通道特征**的“强度”（缩放）和“偏置”（平移）    

:wrench: **实现步骤**（以时间步t为例）  
1. 输入：  
   > :milky_way: **特征图**：来自UNet的某个中间层  
   > :chart_with_upwards_trend: **条件信息**：时间步t —— 通过Embedding得到时间嵌入t_emb（比如通过正弦编码或MLP）
   
2. 将时间嵌入映射为调制参数：
   > 将t_emb通过一个小的MLP（通常是两层全连接），输出形状为：γ, β  
   > 即：对每一个通道c，生成一个scale γ_c 和一个shift β_c
   
   MLP示例：  
   
   ```
   self.mlp = nn.Sequential(
      nn.Linear(time_emb_dim, time_emb_dim),
      nn.ReLU(),
      nn.Linear(time_emb_dim, 2*num_features), # 输出 γ 和 β
   )
   gamma_beta = self.mlp(time_emb) # [B, 2C]
   gamma, beta = torch.chunk(gamma_beta, 2, dim=-1) # 各自【B，C】
   gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
   beta = beta.unsqueeze(-1).unsqueeze(-1)
   ```
4. 应用调制：
   > 对输入特征图 x，逐通道进行：  
   > y = γ · x + β  

   ```
   out = gamma * x + beta
   ```

:arrow_forward: 这样，时间嵌入就通过生成 γ 和 β，控制了该层特征图的“强度”和“偏置”，从而实现了时间条件调制  
:arrow_forward: 即插即用；  
:arrow_forward: 轻量；  
:arrow_forward: 可灵活地扩展到其它条件（如类别、文本嵌入等）  
   
