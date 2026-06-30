# GenZI config 参考

这个目录保存 GenZI 的运行配置。顶层 generation / evaluation 配置通过 `run_cfg=...` 传入，AlphaPose 配置由 generation 配置中的 `pose2d.args_path` 间接引用。

## 数据集和任务背景

GenZI 用同一套 generation / evaluation 管线处理不同来源的 3D 场景。当前仓库提供两组默认配置：

- Sketchfab：来自 Sketchfab 的开放 3D 场景集合，覆盖城市、农场、冬季场景、餐车、健身房等更开放的室内外环境。它主要用于验证 GenZI 在多样、非标准化 3D 资产上的泛化能力。
- PROX-S：来自 PROX / PROX-S 工作流的室内场景集合，场景名称如 `MPH1Library`、`N0SittingBooth`。它更接近人体-室内场景交互评估环境，因此 loss 权重通常更强调物理接触、穿透约束和人体形状正则。

两组配置的整体流程相同：先从 3D scene 渲染多视角图像，再用 Stable Diffusion inpainting 生成 2D 人体交互假设，接着用 AlphaPose 提取 2D 姿态，最后优化 SMPL-X 人体参数并评估语义、物理和多样性指标。差异主要体现在 `group`、`data.scenes`、scene 根目录、初始人体旋转角，以及 PROX-S 更强的物理约束 loss 权重。

## 文件总览

| 文件 | 用途 | 入口命令 |
| --- | --- | --- |
| `sketchfab_gen.yml` | Sketchfab 数据集 generation / optimization | `uv run python -m genzi.generation run_cfg=config/sketchfab_gen.yml` |
| `proxs_gen.yml` | PROX-S 数据集 generation / optimization | `uv run python -m genzi.generation run_cfg=config/proxs_gen.yml` |
| `sketchfab_eval.yml` | Sketchfab 结果 evaluation | `uv run python -m genzi.evaluation run_cfg=config/sketchfab_eval.yml` |
| `proxs_eval.yml` | PROX-S 结果 evaluation | `uv run python -m genzi.evaluation run_cfg=config/proxs_eval.yml` |
| `alphapose.yml` | AlphaPose 子配置，由 generation 的 `pose2d.args_path` 加载 | 不单独作为 GenZI 入口运行 |

## 通用规则

- `path_prefix` 默认为 `./data`，很多路径通过 `${path_prefix}` 展开。
- generation / evaluation 会读取外部 scene config：`data.root_dir/<scene_name><data.cfg_suffix>`。这些 scene config 不在本目录内，通常随数据包放在 `./data/sketchfab` 或 `./data/proxs` 下。
- `gpus` 是 GenZI 顶层 GPU 列表；`alphapose.yml` 中的 `gpus` 是 AlphaPose 自己的参数，格式为字符串。
- generation 配置按 stage 组织：`optim.steps`、`data.view_distances`、`loss.*_weights`、`vlm.dynamic_mask_*` 等列表长度需要和 stage 数量一致。
- `train_ckpt`、`test_ckpt` 当前默认空字符串；generation 主要在每个输出目录下保存并复用 `params.pth`。

## Generation 配置

`sketchfab_gen.yml` 和 `proxs_gen.yml` 结构基本一致，用于 2D inpainting、AlphaPose 2D pose 提取、SMPL-X 参数优化和最终 mesh 输出。

### 顶层运行信息

| 字段 | 当前值 / 差异 | 作用 |
| --- | --- | --- |
| `project` | `GenZI` | wandb / 日志项目名。 |
| `group` | `sketchfab` 或 `proxs` | 决定数据集分组，也会影响 human render 参数分支。 |
| `tags`、`notes` | 空 | 实验元信息。 |
| `gpus` | `[0]` | 使用的 GPU 列表；多 GPU 时 inpainting 会分进程运行。 |
| `seed` | `1` | 随机种子。 |
| `path_prefix` | `./data` | 数据、模型、日志路径前缀。 |
| `log_dir` | `${path_prefix}/log_genzi` | generation 输出根目录。 |
| `ckpt_freq` | `50` | 优化中 checkpoint 保存频率。 |
| `log_freq` | `50` | 优化中图像和标量日志频率。 |
| `save_gif` | `False` | 是否保存优化过程 GIF。 |

### `vposer`

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `ckpt_path` | `${path_prefix}/smpl-x/vposer_V02_05` | VPoser checkpoint 目录。需要手动准备，不会随 `human_body_prior` 自动下载。 |

### `smplx`

| 字段 | 当前值 / 差异 | 作用 |
| --- | --- | --- |
| `model_path` | `${path_prefix}/smpl-x/models_smplx_v1_1` | SMPL-X 模型目录。 |
| `uv_path` | `${path_prefix}/smpl-x/smplx_uv_template.txt` | 导出 textured OBJ 时使用的 UV template。 |
| `tex_path` | `${path_prefix}/smpl-x/smplx_texture_f_alb_1024.png` | 导出 textured OBJ 时使用的 texture。 |
| `model_type` | `smplx` | 传给 `smplx.create`。 |
| `gender` | `neutral` | SMPL-X gender。 |
| `batch_size` | `1` | 单次优化的人体数量。 |
| `num_pca_comps` | `12` | 手部 PCA 分量数。 |
| `use_latent_pose` | `True` | 使用 VPoser latent 优化 body pose。 |
| `use_shape_params` | `True` | 是否优化 SMPL-X shape betas。 |
| `use_continous_rot_repr` | `True` | 使用连续旋转表示优化 orientation / pose。 |
| `rotation_axes` | `xz` | 初始化人体坐标系旋转轴顺序。 |
| `rotation_angles` | Sketchfab: `[90.0, -90.0]`; PROX-S: `[90.0, 90.0]` | 初始化人体坐标系旋转角度，数据集之间不同。 |

### `render`

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `image_size` | `512` | 渲染、inpainting、pose2d 和优化投影使用的图像尺寸。 |

具体光照、背景、材质、相机朝向等 render 参数不在顶层 generation 配置内，而是在每个 scene config 中读取，例如 `render.bg_color`、`render.up_dir`、`render.shadows`。

### `vlm`

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `ldm_inpaint_path` | `stabilityai/stable-diffusion-2-inpainting` | Stable Diffusion inpainting 模型名或本地路径。首次运行会从 Hugging Face cache / Hub 读取。 |
| `neg_prompt_path` | `${path_prefix}/negative_prompts_v1.txt` | 通用 negative prompt 文本文件。 |
| `deterministic` | `False` | 是否固定 inpainting 随机性。 |
| `num_inference_steps` | `50` | diffusion sampling step 数。 |
| `guidance_scale` | `7.5` | classifier-free guidance scale。 |
| `attn_res` | `[16, 16]` | cross-attention map 分辨率。 |
| `attn_thresh_start` / `attn_thresh_end` | `0.7` / `0.7` | 动态 mask attention 阈值范围。 |
| `attn_average_steps` | `3` | 动态 mask 使用的 attention 平均步数。 |
| `dynamic_mask_starts` | `[0, 0]` | 各 stage 开始更新动态 mask 的 step。 |
| `dynamic_mask_stops` | `[25, 0]` | 各 stage 停止更新动态 mask 的 step；第二阶段默认不启用动态更新。 |
| `dilate_size` | `[[0, 0], [11, 11]]` | 各 stage mask dilation kernel。 |
| `dilate_iterations` | `[0, 8]` | 各 stage mask dilation 次数。 |
| `inpaint_dir` | `""` | 非空时复用已有 inpainting 结果；为空时现场生成。 |
| `clip_path` | `openai/clip-vit-base-patch32` | CLIP 模型名或本地路径，用于 semantic scoring。 |

### `pose2d`

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `args_path` | `config/alphapose.yml` | AlphaPose 子配置路径。 |
| `score_thresh` | `0.25` | 2D keypoint 置信度阈值。 |
| `min_num_joints` | `13` | 至少满足该数量的有效 torso/limb joints，才接受一个 inpainting 结果。 |

### `data`

| 字段 | Sketchfab | PROX-S | 作用 |
| --- | --- | --- | --- |
| `root_dir` | `${path_prefix}/sketchfab` | `${path_prefix}/proxs` | 数据集根目录。 |
| `scenes` | 8 个 Sketchfab 场景 | 4 个 PROX-S 场景 | 需要处理的 scene 名称。 |
| `cfg_suffix` | `_v1.yml` | `_v1.yml` | scene config 文件后缀。 |
| `max_views` | `16` | `16` | 每次 interaction 最多使用视角数。 |
| `num_viewpoints` | `256` | `256` | 候选视角采样数量。 |
| `view_distances` | `[2.0, 2.0]` | `[2.0, 2.0]` | 各 stage 视角距离。 |
| `patch_radius` | `0.15` | `0.15` | 用于 scene 表面附近 patch / 法线估计。 |
| `use_at_normal` | `True` | `True` | 采样视角时是否使用 interaction 点附近法线约束。 |
| `fov` | `60` | `60` | 渲染相机 FOV。 |

Sketchfab scenes:

```text
quintyn-glenn-city-scene-kyoto
low-poly-farm-v2
low-poly-winter-scene
a-food-truck-project
bangkok-city-scene
modular_gym
venice-city-scene-1dae08-aaron-ongena
ww2-cityscene-carentan-inspired
```

PROX-S scenes:

```text
MPH1Library
MPH16
N0SittingBooth
N3OpenArea
```

### `optim`

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `type` | `adamw` | 优化器类型。 |
| `grad_clip` | `-1` | 小于等于 0 表示不启用 gradient clipping。 |
| `steps` | 两个 stage，均为 `[400, 400, 400, 400]` | 每个 stage 内 4 段优化步数。 |
| `transl_lrs` | 每段均 `0.01` | SMPL-X translation learning rate。 |
| `orient_lrs` | 每段均 `0.01` | global orientation learning rate。 |
| `pose_lrs` | 每段均 `0.01` | body pose learning rate。 |
| `shape_lrs` | 每段均 `0.01` | shape beta learning rate。 |
| `is_lrs` | `[0.00, 0.01, 0.01, 0.01]` | inpainting score / view selection 参数 learning rate。 |

### `loss`

loss 权重按 stage 和段落组织，通常形如两行列表，每行对应一个 stage，每行里的 5 个数会被插值成 4 段优化权重。

| 字段 | Sketchfab | PROX-S | 作用 |
| --- | --- | --- | --- |
| `inpaint_min_views` | `3` | `3` | 希望至少保留的有效 inpainting 视角数。 |
| `inpaints_per_view` | `[4, 2]` | `[4, 2]` | 每个视角各 stage 生成的 inpainting 数量。 |
| `inpaint_score_weights` | 全 1 | 全 1 | 鼓励保留足够 inpainting 结果。 |
| `joint2d_torso_weights` | 全 1 | 全 1 | torso 2D joint reprojection loss 权重。 |
| `joint2d_limb_weights` | 全 1 | 全 1 | limb 2D joint reprojection loss 权重。 |
| `joint2d_rho` | `0.2` | `0.2` | 2D joint robust loss 参数。 |
| `vposer_weights` | 最高到 `0.001` | 最高到 `0.01` | VPoser latent prior 权重，PROX-S 更强。 |
| `beta0_weight` | `0.5` | `0.5` | 第一个 shape beta 的额外约束权重。 |
| `beta_weights` | 最高到 `0.2` | 最高到 `1.0` | shape regularization 权重，PROX-S 更强。 |
| `scene_intersect_thresh` | `-0.01` | `0` | scene penetration 阈值。 |
| `scene_intersect_weights` | 最高到 `2` | 最高到 `50` | scene intersection penalty，PROX-S 更强。 |
| `scene_nocontact_weights` | 最高到 `10` | 最高到 `1000` | no-contact penalty，PROX-S 更强。 |
| `self_intersect_weights` | 最高到 `1e-5` | 最高到 `1e-4` | human self-intersection penalty。 |
| `angle_weights` | 最高到 `1` | 最高到 `1` | SMPL-X angle prior 权重。 |
| `floating_weights` | 全 0 | 全 0 | floating penalty 当前关闭。 |
| `joint3d_weights` | 全 0 | 全 0 | 3D joint init consistency 当前关闭。 |
| `joint3d_rho` | `1.0` | `1.0` | 3D joint robust loss 参数。 |

## Evaluation 配置

`sketchfab_eval.yml` 和 `proxs_eval.yml` 用于评估已有 generation 结果，不会重新运行 inpainting 或 SMPL-X 优化。

### 共享字段

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `path_prefix` | `./data` | 数据和日志根路径。 |
| `group` | `sketchfab` 或 `proxs` | 数据集分组。 |
| `gpus` | `[0]` | evaluation 使用的 GPU。 |
| `seed` | `1` | 随机种子。 |
| `vposer.ckpt_path` | `${path_prefix}/smpl-x/vposer_V02_05` | 保留在配置中，但 evaluation 当前主要读取生成结果中的 SMPL-X 参数。 |
| `smplx.*` | 与 generation 基本一致 | 记录模型资产路径和模型类型。 |
| `vlm.clip_path` | `openai/clip-vit-base-patch32` | semantic metric 使用的 CLIP。 |
| `render.image_size` | `512` | semantic metric 渲染尺寸。 |
| `metrics.cls_nums` | `[20, 50]` | diversity metric 聚类数量。 |
| `metrics.smplx_params` | `body_pose`, `left_hand_pose`, `right_hand_pose` | diversity metric 使用的 SMPL-X 参数字段。 |
| `data.stages` | `[0, 1]` | 评估哪些 generation stage。 |
| `data.cfg_suffix` | `_v1.yml` | scene config 后缀。 |
| `data.fov` | `60` | evaluation 渲染 FOV。 |

### Sketchfab 与 PROX-S 差异

| 字段 | Sketchfab | PROX-S |
| --- | --- | --- |
| `data.exp_dir` | `${path_prefix}/log_genzi/sketchfab` | `${path_prefix}/log_genzi/proxs` |
| `data.exp_name` | `genzi_sketchfab` | `genzi_proxs` |
| `data.root_dir` | `${path_prefix}/sketchfab` | `${path_prefix}/proxs` |
| `data.include_composition` | `False` | `False` |
| `data.ignore_missing` | `False` | `True` |
| `data.scenes` | 8 个 Sketchfab scenes | 4 个 PROX-S scenes |

`ignore_missing` 的差异很重要：Sketchfab evaluation 默认遇到缺失结果会报错；PROX-S evaluation 默认允许跳过缺失结果。

## AlphaPose 配置

`alphapose.yml` 是 AlphaPose 的参数适配文件，由 `genzi.pose2d.Pose2DPipeline` 加载。代码会先定位已安装的 `alphapose` 包目录，然后把 `cfg` 和 `checkpoint` 拼接为 AlphaPose 安装目录下的相对路径。

| 字段 | 当前值 | 作用 |
| --- | --- | --- |
| `cfg` | `configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml` | AlphaPose 模型结构配置，相对 AlphaPose 安装目录。 |
| `checkpoint` | `pretrained_models/noface_fast50_dcn_combined_256x192.pth` | AlphaPose pose checkpoint，相对 AlphaPose 安装目录。 |
| `sp` | `False` | 是否使用 single process；Windows 下代码会强制设为 `True`。 |
| `detector` | `yolo` | person detector 类型；detector 权重由 AlphaPose 自身配置管理。 |
| `detfile` | `""` | 已有 detection 结果文件。 |
| `inputpath` / `inputlist` / `inputimg` | `""` | GenZI 运行时直接传入图像，通常不需要手动设置。 |
| `outputpath` | `./data/log_alphapose` | AlphaPose 输出目录；GenZI 会覆盖为当前 run 的 `log_dir`。 |
| `save_img` | `True` | 保存 pose 可视化图。 |
| `vis` / `showbox` / `profile` | `False` | 可视化和 profiling 开关。 |
| `format` | `open` | 输出 keypoint 格式。 |
| `min_box_area` | `0` | detector bbox 过滤阈值。 |
| `detbatch` | `5` | detector batch size per GPU。 |
| `posebatch` | `64` | pose model batch size per GPU。 |
| `eval` | `False` | 是否按 COCO evaluation 格式保存。 |
| `gpus` | `"0"` | AlphaPose 自己的 GPU 字符串。 |
| `qsize` | `1024` | 结果队列大小。 |
| `flip` | `False` | 是否启用 flip testing。 |
| `debug` | `False` | 是否输出调试信息。 |
| `video` / `webcam` / `save_video` / `vis_fast` | 默认关闭 | 视频输入和视频保存相关参数，GenZI 当前图像流不使用。 |
| `pose_flow` / `pose_track` | `False` | tracking 开关，GenZI 当前默认不启用。 |

## 修改配置时的检查建议

修改任何 generation 配置后，建议先运行：

```powershell
uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml --no-fail
uv run python tools/check_special_deps.py --run-cfg config/proxs_gen.yml --no-fail
```

如果只调整 evaluation 配置，重点确认：

- `data.exp_dir` 指向已有 generation 输出。
- `data.stages` 和输出目录中的 `stageXXX` 对齐。
- `data.root_dir` 下存在对应 scene config 和 scene mesh / SDF 文件。
- `vlm.clip_path` 可从 Hugging Face cache 或网络读取。
