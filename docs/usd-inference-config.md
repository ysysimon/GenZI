# USD 场景推理与 Config 要求

本文档说明如何使用自己导出的 USD 场景运行 GenZI 推理，包括推荐目录结构、generation config、scene config、必填字段、推理命令和常见排查方式。

## 1. 推荐目录结构

建议把一个自定义 USD 场景相关文件放在同一目录下：

```text
data/
  my_usd_scenes/
    my_room.usd
    my_room_proxy.usd
    my_room.json
    my_room_sdf.npy
    my_room_v1.yml
config/
  my_usd_gen.yml
```

其中：

- `my_room.usd` 是你导出的 USD 场景。
- `my_room_proxy.usd` 是可选的轻量视角采样几何。大场景建议准备，仍然使用 USD 格式。
- `my_room.json` 和 `my_room_sdf.npy` 是场景 SDF。当前 `Scene` 初始化一定会读取 SDF，因此不是可选项。
- `my_room_v1.yml` 是单个 scene config。
- `config/my_usd_gen.yml` 是顶层 generation config，用来指定要跑哪些 scene。

如果需要单独的视角采样几何，也建议继续使用 USD 文件，并在 `scene.subd_mesh_path` 指向对应的 `.usd`。室内场景的采样几何应保留墙、天花板、地面和主要家具，否则自动采样视角可能会穿墙或选到不合理视角。对于千万级 faces 的 USD，不建议直接把完整 mesh 用作 `subd_mesh_path`，否则 `nvdiffrast` 视角筛选可能报 CUDA error。

## 2. 顶层 Generation Config

可以复制 `config/sketchfab_gen.yml`：

```powershell
Copy-Item config\sketchfab_gen.yml config\my_usd_gen.yml
```

然后重点修改 `data` 段：

```yaml
group: sketchfab

data:
  root_dir: "./data/my_usd_scenes"
  scenes:
    - "my_room"
  cfg_suffix: "_v1.yml"
  max_views: 16
  num_viewpoints: 256
  view_distances:
    - 2.0
    - 2.0
  patch_radius: 0.15
  use_at_normal: true
  fov: 60
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `group` | 数据分支。自定义 USD 场景可以先沿用 `sketchfab`。 |
| `data.root_dir` | scene config 所在目录。 |
| `data.scenes` | 要运行的 scene 名称列表。每个名称会和 `cfg_suffix` 拼成 scene config 文件名。 |
| `data.cfg_suffix` | scene config 后缀。例如 `my_room` + `_v1.yml` = `my_room_v1.yml`。 |
| `data.max_views` | 每次 interaction 最多保留多少个有效视角。 |
| `data.num_viewpoints` | 自动采样候选视角数量。 |
| `data.view_distances` | 各 stage 的相机到交互点距离。长度要和 `optim.steps` 的 stage 数一致。 |
| `data.patch_radius` | 交互点附近 patch 搜索半径，用于估计局部可见区域和法线。 |
| `data.use_at_normal` | 是否用交互点附近法线约束候选视角。 |
| `data.fov` | 渲染相机 FOV。 |

## 3. Scene Config 模板

新建 `data/my_usd_scenes/my_room_v1.yml`：

```yaml
scene:
  mesh_path: "./data/my_usd_scenes/my_room.usd"
  sdf_path: "./data/my_usd_scenes/my_room.json"
  subd_mesh_path: "./data/my_usd_scenes/my_room_proxy.usd"

  usd_prim_path: null
  usd_purpose: render
  usd_include_invisible: false
  usd_time_code: null

render:
  bg_color: [0.5, 0.5, 0.5, 0.0]
  ambient_light: [0.25, 0.25, 0.25]
  dir_light_color: [1.0, 1.0, 1.0]
  dir_light_intensity: 3.0
  pt_light_color: [1.0, 1.0, 1.0]
  pt_light_intensity: 1.0
  pt_light_position: [0.0, 0.0, 2.5]
  normal_pbr: true
  no_lighting: false
  all_solid: false
  cull_faces: false
  shadows: false
  up_dir: [0.0, 0.0, 1.0]

prompt_prefix: ""
prompt_suffix: ""
token_indices: [-1]

prompt_ids:
  - "sit_chair_001"

prompts:
  - "a person sitting on the chair"

neg_prompts:
  - ""

interactions:
  - "sit"

lookats:
  - [1.2, 0.8, 0.45]

viewpoints:
  - null
```

## 4. Scene 字段要求

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `scene.mesh_path` | 是 | 主场景 mesh。你的导出文件通常写 `.usd`。最终 scene render、depth 遮挡和 human-scene compositing 都会用它。 |
| `scene.sdf_path` | 是 | 场景 SDF 路径。可以写 `.json` 或 `_sdf.npy`，但对应的两个文件都需要存在。 |
| `scene.subd_mesh_path` | 是 | 视角采样用 mesh。写空字符串时直接使用 `scene.mesh_path`。大 USD 场景建议指向轻量 `.usd` proxy，并保留墙、天花板、地面和主要遮挡物。 |

SDF 命名规则：

- 如果 `sdf_path` 写 `my_room.json`，代码会读取 `my_room.json` 和 `my_room_sdf.npy`。
- 如果 `sdf_path` 写 `my_room_sdf.npy`，代码会读取 `my_room_sdf.npy` 和 `my_room.json`。

## 5. USD 读取字段

这些字段用于读取 `.usd` 场景；本文档里的自定义场景流程默认都按 `.usd` 编写。

| 字段 | 推荐值 | 含义 |
| --- | --- | --- |
| `scene.usd_prim_path` | `null` | 只读取某个 absolute prim path 下的 subtree。例如 `/World/Room`。`null` 表示读取整个 stage。 |
| `scene.usd_purpose` | `render` | USD purpose 过滤。可选 `default`、`render`、`proxy`、`all`。如果读不到 mesh，先改成 `all` 排查。 |
| `scene.usd_include_invisible` | `false` | 是否读取 invisible mesh prim。 |
| `scene.usd_time_code` | `null` | 动画 USD 的采样 time code。静态场景写 `null`。 |

注意：当前 USD loader 会合并 `UsdGeom.Mesh` prim，但不会保留 USD material、texture、UV、normal、instance、variant composition 等高层信息。

## 6. Render 字段建议

室内场景推荐先使用：

```yaml
ambient_light: [0.25, 0.25, 0.25]
dir_light_intensity: 3.0
pt_light_intensity: 1.0
shadows: false
```

这样仍然保留方向光和点光源带来的明暗层次，但墙和天花板不会作为 shadow caster 把画面遮黑。

| 字段 | 含义 |
| --- | --- |
| `render.bg_color` | 背景色，RGBA。 |
| `render.ambient_light` | 环境光，提供基础亮度，不产生方向阴影。 |
| `render.dir_light_color` | 方向光颜色。当前实现中方向光会跟随 camera pose，更像 key light。 |
| `render.dir_light_intensity` | 方向光强度。 |
| `render.pt_light_color` | 点光源颜色。 |
| `render.pt_light_intensity` | 点光源强度。 |
| `render.pt_light_position` | 点光源世界坐标。 |
| `render.normal_pbr` | 是否使用带光照的 PBR 渲染路径。一般设为 `true`。 |
| `render.no_lighting` | 是否禁用光照。一般设为 `false`。 |
| `render.all_solid` | 是否强制 solid 渲染。一般设为 `false`。 |
| `render.cull_faces` | 是否剔除背面。室内场景通常设为 `false`。 |
| `render.shadows` | 是否开启 shadow render flags。室内场景如果担心墙和天花板挡光，设为 `false`。 |
| `render.up_dir` | 世界坐标上方向。Z-up 用 `[0.0, 0.0, 1.0]`，Y-up 用 `[0.0, 1.0, 0.0]`。 |

## 7. Prompt 与 Interaction 字段

这些字段按 interaction 一一对应。也就是说，`prompt_ids`、`prompts`、`neg_prompts`、`interactions`、`lookats`、`viewpoints` 的列表长度必须一致。

| 字段 | 含义 |
| --- | --- |
| `prompt_prefix` | 每个 prompt 前统一拼接的文本。 |
| `prompt_suffix` | 每个 prompt 后统一拼接的文本。 |
| `token_indices` | inpainting 动态 mask 关注的 prompt token 下标。`[-1]` 表示最后一个 token，可先用于试跑。 |
| `prompt_ids` | 每个任务的 ID，用于输出目录命名。 |
| `prompts` | 交互文本描述，例如 `a person sitting on the chair`。 |
| `neg_prompts` | 单个 prompt 对应的 negative prompt。可以先写空字符串。 |
| `interactions` | 交互标签，例如 `sit`、`stand`、`lie`。主要用于记录和输出。 |
| `lookats` | 最重要的交互点 3D 坐标。例如椅子座面中心、床面中心、桌前地面点。 |
| `viewpoints` | 手动指定相机 eye 列表。写 `null` 时自动采样，通常建议先用 `null`。 |

多个 interaction 示例：

```yaml
prompt_ids:
  - "sit_chair_001"
  - "stand_table_001"

prompts:
  - "a person sitting on the chair"
  - "a person standing in front of the table"

neg_prompts:
  - ""
  - ""

interactions:
  - "sit"
  - "stand"

lookats:
  - [1.2, 0.8, 0.45]
  - [2.0, 1.1, 0.0]

viewpoints:
  - null
  - null
```

## 8. 推理前检查

先确认 USD 能被读取：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_usd_scenes\my_room.usd `
  --out data\my_usd_scenes\my_room_preview.png `
  --purpose render `
  --style solid-wire
```

如果没有读到 mesh，放宽过滤条件：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_usd_scenes\my_room.usd `
  --out data\my_usd_scenes\my_room_debug_all.png `
  --purpose all `
  --include-invisible `
  --style solid-wire
```

如果只想读取某个 subtree：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_usd_scenes\my_room.usd `
  --out data\my_usd_scenes\my_room_subtree.png `
  --prim-path /World/Room `
  --purpose render `
  --style solid-wire
```

如果完整 USD 很大，可以先生成一个同样是 `.usd` 的视角采样 proxy：

```powershell
uv run python tools\build_usd_proxy.py `
  data\my_usd_scenes\my_room.usd `
  data\my_usd_scenes\my_room_proxy.usd `
  --voxel-size 0.06 `
  --max-cells 200000 `
  --purpose default
```

然后在 scene config 里设置：

```yaml
scene:
  subd_mesh_path: "./data/my_usd_scenes/my_room_proxy.usd"
```

## 9. 推理命令

基础推理命令：

```powershell
uv run python genzi\generation.py run_cfg=config\my_usd_gen.yml
```

常用覆盖参数：

```powershell
uv run python genzi\generation.py `
  run_cfg=config\my_usd_gen.yml `
  gpus=[0] `
  log_dir=./data/log_my_usd
```

如果只想临时跑某一个 scene：

```powershell
uv run python genzi\generation.py `
  run_cfg=config\my_usd_gen.yml `
  data.scenes=[my_room]
```

## 10. 常见问题

### nvdiffrast Cuda error: 700

常见原因是 `scene.subd_mesh_path` 直接使用了过大的完整 USD mesh。解决方式是生成轻量 `.usd` proxy，并让 `subd_mesh_path` 指向它。proxy 仍然要保留室内场景的墙、天花板、地面和主要家具，否则视角采样会失真。

### AlphaPose detector 找不到 cfg 或 weights

需要确认以下文件存在：

```text
external/alphapose/detector/yolo/cfg/yolov3-spp.cfg
external/alphapose/detector/yolo/data/yolov3-spp.weights
```

如果缺少 `yolov3-spp.weights`，按 AlphaPose 文档下载后放到 `external/alphapose/detector/yolo/data/`。

### negative prompt 文件缺失

generation config 里的 `vlm.neg_prompt_path` 默认指向：

```yaml
vlm:
  neg_prompt_path: "${path_prefix}/negative_prompts_v1.txt"
```

因此需要存在 `data/negative_prompts_v1.txt`，否则初始化 scene 时会直接报 `FileNotFoundError`。

### No usable UsdGeom.Mesh prims found

常见原因：

- `scene.usd_purpose` 过滤太严格。先试 `usd_purpose: all`。
- mesh 是 invisible，但 `scene.usd_include_invisible` 是 `false`。
- `scene.usd_prim_path` 写错，或者该 subtree 下没有 `UsdGeom.Mesh`。
- 指定 `scene.usd_time_code` 时，该时间没有有效 points 或 face indices。

### 画面太黑

室内场景先使用：

```yaml
ambient_light: [0.25, 0.25, 0.25]
dir_light_intensity: 3.0
pt_light_intensity: 1.0
shadows: false
```

不要只依赖 `ambient_light`。方向光和点光源仍然可以保留，用来提供明暗层次。

### 自动视角不合理

优先检查：

- `lookats` 是否真的落在目标交互位置附近。
- `render.up_dir` 是否和 USD 坐标系一致。
- `scene.subd_mesh_path` 是否保留了墙、天花板、地面和主要遮挡物。
- `data.view_distances` 是否适合房间大小。
- `data.use_at_normal` 是否需要关闭试跑。

### SDF 报错或穿模约束异常

确认 `scene.sdf_path` 对应文件成对存在：

```text
my_room.json
my_room_sdf.npy
```

并确认 SDF 的坐标系、尺度和 USD 场景一致。
