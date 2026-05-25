# checkpoints/

训练产物目录，所有内容已被 `.gitignore` 忽略。

`python main.py train` 会在此写入：

- `best_model/` — 验证集表现最好的 LoRA 适配器
- `checkpoint_epoch_<N>/` — 每个 epoch 的快照

每个子目录包含 `adapter_config.json`、`adapter_model.safetensors`、`training_state.pt` 等。
