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
   维度与模型中间层的通道数相匹配，以便与图像特征融合
   
3. **Time Embedding如何注入UNet？**
4. **Time Embedding注入UNet后，如何调整其对不同时间步的响应？**
