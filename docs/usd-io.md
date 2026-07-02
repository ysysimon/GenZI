# USD IO 使用说明

本文记录 GenZI 当前的 USD scene 读取、转换和预览用法。相关实现主要在 `genzi/usd_io.py`，辅助命令在 `tools/convert_usd_scene.py` 和 `tools/preview_mesh_scene.py`。

## 支持范围

当前支持读取 `.usd`、`.usda`、`.usdc` 文件中的 `UsdGeom.Mesh` prim，并把它们合并成一个 `trimesh.Trimesh`，供 GenZI 的 scene 渲染、generation 和 evaluation 流程继续使用。

读取时会做以下处理：

- 遍历 USD stage，或只遍历 `--prim-path` / `scene.usd_prim_path` 指定的 subtree。
- 只读取 `UsdGeom.Mesh` prim。
- 默认跳过 invisible mesh；需要保留时设置 `include_invisible`。
- 默认只读取 `purpose=default` 的 mesh；也可以选择 `render`、`proxy` 或 `all`。
- 按指定 `time_code` 采样 points、face indices 和 transform。
- 把 prim 的 local-to-world transform 烘焙到顶点坐标里。
- 把 quad 和 ngon 三角化。
- 合并多个 mesh prim 为一个 `trimesh.Trimesh`。

当前不会读取或保留 USD material、texture、UV、normal、instance、variant composition 等高层 USD 信息。`metersPerUnit` 只作为 metadata 记录在 `load_usd_mesh_with_metadata()` 的 stats 中，不会自动缩放顶点。

## 依赖

USD IO 依赖：

```text
usd-core==26.3
trimesh[easy]
```

PNG preview 额外依赖：

```text
matplotlib==3.7.1
```

仓库的 `pyproject.toml` 和 `requirements.txt` 已经包含这些依赖。本地 uv 环境通常直接使用：

```powershell
uv run python -c "from pxr import Usd, UsdGeom; import trimesh; print('usd io ok')"
```

## Scene Config 用法

scene config 可以直接把 `scene.mesh_path` 指向 USD 文件。GenZI 在 generation 和 evaluation 中会通过 `genzi.misc.get_scene_usd_options()` 读取 USD 选项，然后传给 scene mesh loader。

示例：

```yaml
scene:
  mesh_path: ./data/my_scene/scene.usd
  sdf_path: ./data/my_scene/sdf.npy
  subd_mesh_path: ./data/my_scene/scene_proxy.usd

  usd_prim_path: /World/Room/ChairArea
  usd_purpose: default
  usd_include_invisible: false
  usd_time_code: 12.0
```

可选字段说明：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `scene.usd_prim_path` | 空 | 只读取某个 absolute prim path 下的 subtree，也可以直接指向 mesh prim。 |
| `scene.usd_purpose` | `default` | 可选 `default`、`render`、`proxy`、`all`。`render` / `proxy` 会同时包含 default mesh。 |
| `scene.usd_include_invisible` | `false` | 是否包含 invisible mesh prim。 |
| `scene.usd_time_code` | USD default time | 指定采样的 USD time code。 |

如果 `scene.mesh_path` 不是 `.usd`、`.usda` 或 `.usdc`，这些 USD 字段会被忽略，仍按普通 `trimesh.load()` 流程读取。

## Python API

只需要 mesh 时：

```python
from genzi.usd_io import UsdMeshLoadOptions, load_usd_mesh

mesh = load_usd_mesh(
    "data/my_scene/scene.usd",
    options=UsdMeshLoadOptions(
        prim_path="/World/Room/ChairArea",
        purpose="render",
        include_invisible=False,
        time_code=12.0,
    ),
)
print(len(mesh.vertices), len(mesh.faces))
```

需要统计信息时：

```python
from genzi.usd_io import UsdMeshLoadOptions, load_usd_mesh_with_metadata

result = load_usd_mesh_with_metadata(
    "data/my_scene/scene.usd",
    options=UsdMeshLoadOptions(purpose="all"),
)

mesh = result.mesh
stats = result.stats
print(stats.mesh_count)
print(stats.skipped_invisible_count)
print(stats.skipped_purpose_count)
print(stats.skipped_empty_count)
print(stats.stage_meters_per_unit)
```

## 转换为普通 Mesh

这一步只用于诊断和几何检查，不是 GenZI 推理的推荐输入格式。推理配置里仍建议直接使用你导出的 `.usd`，或为 `scene.subd_mesh_path` 准备轻量 `.usd` proxy。

```powershell
uv run python tools\convert_usd_scene.py `
  --usd data\my_scene\scene.usd `
  --out data\my_scene\scene_preview.ply `
  --prim-path /World/Room/ChairArea `
  --purpose render `
  --print-summary
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--usd` | 输入 `.usd`、`.usda` 或 `.usdc` 文件。 |
| `--out` | 诊断用 mesh 输出路径，例如 `.ply`。实际格式由 `trimesh` 根据后缀导出。 |
| `--prim-path` | 只转换指定 prim path 下的 subtree。 |
| `--purpose` | `default`、`render`、`proxy` 或 `all`。 |
| `--include-invisible` | 包含 invisible mesh prim。 |
| `--time-code` | 指定 USD time code。 |
| `--print-summary` | 输出 mesh 数量、顶点数、面数、bbox、跳过统计和 stage unit metadata。 |

## Preview

`tools/preview_mesh_scene.py` 可以快速检查 USD 或普通 mesh 的几何范围和大致形状。它会打印 mesh summary；如果传入 `--out`，还会保存一个 2x2 视角的 PNG，包含 iso、front、side、top 四个视角。

最常用命令：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_scene\scene.usd `
  --out previews\scene_preview.png
```

只预览某个 subtree：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_scene\scene.usd `
  --out previews\chair_area.png `
  --prim-path /World/Room/ChairArea
```

使用 solid-wire 风格并保留 render purpose：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_scene\scene.usd `
  --out previews\scene_render.png `
  --purpose render `
  --style solid-wire `
  --max-faces 100000
```

只看 summary，不导出图片：

```powershell
uv run python tools\preview_mesh_scene.py --mesh data\my_scene\scene.usd
```

打开交互 viewer：

```powershell
uv run python tools\preview_mesh_scene.py --mesh data\my_scene\scene.usd --interactive
```

Preview 参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mesh` | 必填 | 输入 USD 或 `trimesh` 支持的 mesh 文件。 |
| `--out` | 空 | 输出 PNG 路径。为空时只打印 summary，除非设置 `--interactive`。 |
| `--interactive` | false | 打开 `trimesh` 交互 viewer。 |
| `--prim-path` | 空 | USD subtree 过滤；非 USD mesh 会忽略。 |
| `--purpose` | `default` | USD purpose 过滤。 |
| `--include-invisible` | false | 包含 invisible USD mesh prim。 |
| `--time-code` | USD default time | 指定 USD time code。 |
| `--max-faces` | `50000` | PNG 中最多绘制的 face 数。小于等于 0 表示不限制。 |
| `--sample-by` | `area` | 超过 `--max-faces` 时的采样方式：`area` 保留面积最大的面，`stride` 等距抽样，`random` 随机抽样。 |
| `--style` | `wireframe` | PNG 样式：`wireframe`、`solid`、`solid-wire`。 |
| `--up-axis` | `y` | 源 mesh 的 up axis，会映射到 preview 的竖直方向。 |
| `--seed` | `1` | `--sample-by random` 时使用的随机种子。 |

Preview 主要用于快速检查几何是否读对、坐标轴是否合理、`prim_path` / `purpose` 过滤是否符合预期。它不是高保真 USD viewport，不会展示材质、贴图或灯光。

## 常见问题

### 报错：`USD mesh conversion requires the pxr module`

当前环境没有可导入的 `pxr` 模块。确认在 GenZI 环境里安装了 `usd-core==26.3`：

```powershell
uv run python -c "from pxr import Usd; print(Usd)"
```

### 报错：`USD prim path must be an absolute prim path`

`prim_path` 必须以 `/` 开头，例如 `/World/Room/ChairArea`，不能写成 `World/Room/ChairArea`。

### 报错：`USD prim path not found in stage`

指定的 prim path 在 stage 中不存在。可以先不传 `--prim-path` 跑一次 preview summary，或在 USD 工具中查看 stage 结构后再指定。

### 报错：`No usable UsdGeom.Mesh prims found`

常见原因：

- 指定 subtree 下没有 `UsdGeom.Mesh`。
- mesh 是 invisible，但没有设置 `--include-invisible` / `scene.usd_include_invisible: true`。
- mesh 的 purpose 不是当前过滤条件包含的类型。
- mesh 在指定 time code 下没有 points 或 face indices。

可以临时放宽过滤条件检查：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_scene\scene.usd `
  --out previews\debug_all.png `
  --purpose all `
  --include-invisible
```

### Preview PNG 看起来方向不对

尝试调整 `--up-axis`。默认值是 `y`，适合 Y-up 的源 mesh。如果 USD asset 是 Z-up，可以使用：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_scene\scene.usd `
  --out previews\scene_z_up.png `
  --up-axis z
```

### Preview 太慢或图片太密

降低 `--max-faces`，或者换采样方式：

```powershell
uv run python tools\preview_mesh_scene.py `
  --mesh data\my_scene\scene.usd `
  --out previews\scene_fast.png `
  --max-faces 20000 `
  --sample-by stride
```

## 测试

USD IO 和 preview CLI 的覆盖测试在 `tests/test_usd_io.py`：

```powershell
uv run pytest tests\test_usd_io.py
```
