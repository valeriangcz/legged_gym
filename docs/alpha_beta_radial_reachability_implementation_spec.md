# 基于安全球壳的 alpha-beta 径向分区与方向归属

## 1. 当前实现目标

在 `env_robot_voxels.py` 的现有框架内，为固定机身位姿建立六条腿各自的二维方向栅格。
每个方向保存两个 IK 分支的径向连续前缀上界；只有从安全半径到外侧要求边界的整段采样均有效，才参与连通分区。

空间构建与落脚点判断是两套独立逻辑：先完成方向分区，再使用原有可行落脚点和分支掩码，按方向统计各区域的落脚点。
不实现 `QuerySections()`，不根据落脚点半径筛选，不修改现有规划器的筛选行为。

本文替代此前的“区间重叠加局部运动检查”方案，以安全球壳假设为准。

## 2. 坐标和角度约定

所有球面运算在每条腿自己的基坐标系 B 中进行。世界系、身体系和腿基系之间复用现有 `Kinematic._R2B()`、`_B2R()` 和 `SE3` 变换，保留左右腿轴向差异。

保留当前代码的命名：**alpha 为方位角，beta 为仰角**。这与早期文档的字母分配不同。

\[
r=\sqrt{x^2+y^2+z^2},\qquad
\alpha=\operatorname{atan2}(y,x),\qquad
\beta=\operatorname{atan2}(z,\sqrt{x^2+y^2}).
\]

\[
u(\alpha,\beta)=
[\cos\beta\cos\alpha,\ \cos\beta\sin\alpha,\ \sin\beta]^T,
\qquad p=r u.
\]

球面栅格接口、存储的角度及角度步长统一使用弧度，不在正反变换中进行度数转换。
每条腿 alpha 范围直接使用 thigh 的弧度限位，beta 范围为 [-pi/2,pi/2]。
alpha 的支持范围为 [-pi,pi]。需要按度数配置时，在调用处使用 `np.deg2rad()`。

## 3. 安全球壳与 3 cm 径向带

选择固定的正收拢半径 r0。安全球壳指机身附近允许腿部收拢的区域，不意味着连杆进入机身实体。
采用以下角向简化：同一 IK 分支的相邻有效方向可以沿 r0 球壳连续转向，不额外执行角向运动或关节插值检查。

默认检查从 r0 到 r0 + 0.03 m 的径向带，复用机器人体素尺寸作为采样步长，当前默认约 1 cm。
例如 r0 = 0.145 m 时，检查 0.145、0.155、0.165、0.175 m 四层。
一次构造所有半径和方向的采样点，批量查询机器人表和环境碰撞，不再逐半径循环。
仍保留首次失败之前的前缀半径，供诊断使用：

\[
I_{ij,b}=[r_0,r_{\max,ij,b}].
\]

- 在批量结果上沿半径轴执行前缀逻辑与，任意位置失败后，即使更远处可行也不恢复。
- 两个分支独立判断；某个方向的一个分支不影响另一个分支或其他方向。
- 起点失败时，上界为 -inf；中途失败时，上界为最后一个连续有效的采样半径，但方向仍不参与分区。
- 采样包含 r0 和要求的外侧半径，最后一步可短于径向步长。
- 只有 r_max >= required_radius 的方向/分支组合参与分区，即整段径向采样全部通过。
- 使用批量检查后不再提前终止碰撞查询；首次失败截断通过结果归约实现。
- 不要求射线与环境只有一个交点，也不另建环境交点距离表；使用首次检测到的失败截断。

这里的安全球壳转向是建模假设。格子中心或相邻角度足够接近，本身不构成连续运动无碰撞的证明。
默认 r0 是可配置初值，不代表已经证明整层球壳安全。径向采样全部通过也不等于连续扫掠或固定碰撞间隙证明。

## 4. 类职责及数据布局

### SphereVoxels

与规则 xyz `Voxels` 并列，不继承其索引逻辑。
负责角度栅格、球坐标正反变换、方向索引、双分支区间上界、区域标记和落脚点统计。

```text
bounds:          (2,2)，alpha/beta 的弧度范围
scale:           角度步长，单位弧度
grid_shape:     (n_alpha,n_beta)
grid_size:      n_alpha*n_beta
angle_centers:   (n_alpha,n_beta,2)
directions:      (grid_size,3)，展平的单位方向
radius_grid:     (n_alpha,n_beta,2)，首次失败前的连续上界，起点失败为 -inf
section_grid:    (n_alpha,n_beta,2)，整数，无效为 -1
section_count:   (2,)，各分支区域数量
safe_radius:     最近一次构建使用的 r0；尚未构建时为 None
required_radius: 要求整段通过的外侧半径，默认 r0 + 0.03 m
```

角度格使用中心方向。范围不能被步长整除时，最后一格按实际边界截断，并使用截断后中心。
采用 `meshgrid(indexing="ij")` 和 C-order 展平，保持所有数组对应关系一致。
角度上下边界均可查询，精确上界归入最后一格。无效方向不裁剪入有效格；首尾不周期连接。

### LegVoxels / RoboVoxels / HexState

`LegVoxels.sphere_voxels_list` 保存六张表；初始化时创建固定方向结构。
`RoboVoxels`、`HexState` 接受可选 `sphere_scale` 并向下传递。

`HexState._LegConnections()` 构建单腿结果，`BuildRobotConnections()` 为给定位姿依次构建六张表。
位姿平移、旋转或环境变化后重新调用；方向栅格复用，区间与标签清空后重建。
返回的是保存最新结果的表对象，下次构建会原位更新，不是历史快照。不写入离线缓存。

## 5. 碰撞与有效性

复用现有 `_LegEnvCollisionFree()`，启用 ankle 和足端两项检查：

- 离线机器人可达掩码包含已有的关节限位和机器人本体约束；
- ankle 使用现有半径阈值，默认 0.045 m；
- 足端使用现有环境 ESDF 阈值 0.007 m。

构建路径额外检查采样点是否在机器人表内、IK/ankle 数据是否有限、采样末端及体素中心末端和 ankle 是否在环境地图内。
越界判为失败，避免现有裁剪索引将地图外位置映射到自由边界。
现有碰撞函数其他调用路径保持不变。

本版不新增完整 thigh/shank 碰撞、腿间碰撞、关节连续性传播或扫掠检查。
机器人表、地图离散分辨率及现有碰撞模型共同决定结果精度。

## 6. 连通分区

只将径向上界达到 required_radius 的方向视为有效，有限上界不再自动代表方向有效。
所有有效区间都包含完整的 [r0,required_radius]，因此不再比较相邻区间是否重叠。
在安全球壳假设下，对每个分支的有效方向掩码分别执行四邻域 `scipy.ndimage.label()`。
区域编号在每条腿、每个分支内从 0 开始，完整标识为 (leg_index,branch,section_id)。

对角格、角度首尾不直接连接；两个分支永不合并。
如果某分支整个有效角域连通，则该分支只有一个区域，这是预期结果。

## 7. 接口

```python
SphereVoxels(bounds, scale)  # bounds 和 scale 均为弧度
Pos2Sphere(pos)          # (x,y,z) -> (alpha,beta,r)
Sphere2Pos(sphere_pos)   # (alpha,beta,r) -> (x,y,z)
Pos2GridIndex(pos)       # 按方向返回索引，无效为 (-1,-1)
LabelSections()         # 按 required_radius 筛选完整厚度，再分别标记两个分支
Pos2Section(B_points)   # 按方向返回标签，不比较半径
CountFootholds(B_points, feasible_mask)  # 返回标签和各分支计数

HexState._LegConnections(
    W_T_R, leg_index, *, safe_radius=0.145,
    max_radius=None, radial_step=None,
)
HexState.BuildRobotConnections(
    W_T_R, *, safe_radius=0.145,
    max_radius=None, radial_step=None,
)
```

`max_radius=None` 时，外边界自动取 `safe_radius + 0.03`；显式指定时，要求整段通过到该半径。
`safe_radius == max_radius` 仍可显式用于单层诊断，但不是默认初筛模式。
手动构造的 `SphereVoxels` 在尚未设置 `required_radius` 时，`LabelSections()` 保留按有限上界分区的行为。

坐标转换、索引和方向归属接受单点 `(3,)` 或批量 `(N,3)`，输出相应的单点或批量形状。
零向量、非有限点、角度越界点归属无效。

`CountFootholds()` 的 `feasible_mask` 必须为 `(N,2)`，返回：

- `labels`：`(N,2)`，方向无区域或对应可行分支为 False 时取 -1；
- `counts`：两个分支的计数数组，数组长度分别等于各自区域数量，包含零落脚点区域。

同一点若两个分支都可行，分别计入两个分支。同方向不同正半径得到相同区域归属，哪怕该半径超出区间；这仅表示方向归属，不替代落脚点或轨迹可行性判断。
不需要法向偏移、预接触点或接近动作检查。

## 8. 使用顺序

```python
state = HexState(Kinematic(), sphere_scale=np.deg2rad(5.0))  # 粗方向初筛
tables = state.BuildRobotConnections(W_T_R)
_, landing_idx, _, feasible = state.RobotFeasiCheck(W_T_R)
landing_world = state.env_pointsmap_voxels.landing_points[landing_idx]
landing_robot = (W_T_R.inv() * landing_world.T).T

for leg_index, table in enumerate(tables):
    landing_base = state.kin._R2B(landing_robot.T, leg_index).T
    labels, counts = table.CountFootholds(landing_base, feasible[leg_index])
    has_footholds = [branch_counts > 0 for branch_counts in counts]
```

两套逻辑显式、独立调用，现有 `RobotFeasiCheck()` 返回值及规划器筛选行为不变。

## 9. 默认参数与验证

- 收拢半径：0.145 m；要求外半径默认 0.175 m（收拢半径 + 3 cm）。
- 角度步长：`np.deg2rad(1.0)`，即 pi/180 rad，对应 1 度；径向步长默认使用机器人体素尺寸，当前默认约 0.01 m。
- 四邻域，不周期连接，两个分支各自分区。
- 超出机器人表或环境地图时判为无效。

无 GUI 测试位于 `test_sphere_voxels.py`，使用小型内存地图和合成碰撞结果，不重建机器人缓存。
覆盖坐标和边界、整段厚度有效性、单次批量查询、不可恢复前缀、与逐层参考实现一致、外半径端点、足端/ankle 碰撞、越界、重复构建、四邻域分区、仅按方向统计及零落脚点区域。

```bash
python3 -m unittest legged_gym.expert_complex_utils.test_sphere_voxels
python3 -m unittest legged_gym.expert_complex_utils.test_mechanical_margin
```

手动三面板可视化：安全收拢层、径向上界、区域与落脚点。
`--sphere-scale` 使用弧度，默认 pi/180；例如 `--sphere-scale 0.03490658503988659` 对应 2 度。
`--max-radius` 默认使用收拢半径 + 3 cm；可显式指定其他要求外半径。
xyz 是世界位置，演示入口中独立的机身 rpy 参数仍使用度；需要根据实际地图选择位姿。

```bash
python3 -m legged_gym.expert_complex_utils.test_val --connections --leg 2 --branch 0 --xyz 0 0 0 --rpy 0 0 0
```

不带 `--connections` 时保留原有单腿工作空间可视化，默认显示腿 2。
