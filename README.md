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
   
2. **Time Embedding如何注入UNet？**  
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
   
   
   
   
5. **Time Embedding注入UNet后，如何调整其对不同时间步的响应？**
