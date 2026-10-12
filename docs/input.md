# 输入与相机坐标

```text
INPUT/
  场景.pro
  场景_CameraPos_C2E.txt
  场景_pos.txt
  场景_pos_ENH.txt
  Image/L/**/*.JPG
  Image/R/**/*.JPG
  ResultOut/场景.las
```

仅使用 `.pro` 中 L/R 相机及 CameraPos 有精确同名记录的 JPEG，M 不参与。图像名和各镜头帧号须唯一，Pose 绑定图像缺失时报错。尺寸须等于 `.pro`，EXIF orientation 须为 1。按 GPS 周、文件名时间、镜头和图像名排序；L/R 同帧号不表示同步。

文件名为 `GPS周-秒-毫秒_帧号-L或R.JPG`。CameraPos 每行 11 字段：文件名、设备编号、GPS 周、秒、XYZ、四元数四分量。仅使用位置，四元数不用于视觉初始化；不改写时间戳。

鱼眼参数为 `fx fy cx cy k1 k2 k3 k4`。初值：`fx=fy=focalLength/pixelSize`，`cx=width/2+principalX/pixelSize`，`cy=height/2-principalY/pixelSize`，`k4=p3`。继承该设备映射，厂商主点方向和 p3 含义尚未独立核对。

左上像素中心为 `(0.5,0.5)`。`theta=atan2(sqrt(X²+Y²),Z)`，径向多项式为 `theta*(1+k1*theta²+k2*theta⁴+k3*theta⁶+k4*theta⁸)`。有效域为完整视场 160°，多项式须单调。L/R 内参各共享一组，每次曝光独立位姿。

两份导航表行身份须一致，前五列为 GPS 周、秒、XYZ，时序递增。由成对位置拟合 ECEF→ENH 转换：尺度偏离 1 不超过 0.001、逐行残差不超过 2 cm。CameraPos 位置经该转换形成导航先验。

LAS 支持未压缩 1.2、点格式 0–3；多份使用 `--las` 选择。排除 withheld 和 class-7 噪声，RGB 支持 uint16 中的 8-bit/16-bit 范围，无 RGB 时灰色 127。

配准继承 `LAS XYZ ≈ ENH 减首个 ENH 位置` 假设，单位按米解释；拟合不能确认绝对 CRS 或独立地理精度。计算使用局部原点，输出恢复输入 LAS 坐标。
