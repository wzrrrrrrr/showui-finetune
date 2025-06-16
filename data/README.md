# 数据集说明

## 目录结构
```
data/
├── README.md           # 本说明文件
├── metadata.jsonl      # 数据集元数据文件
└── my_dataset/         # 存放训练图片的目录
    ├── example1.png
    ├── example2.png
    └── ...
```

## 数据格式说明

### metadata.jsonl 格式
每行是一个JSON对象，包含以下字段：
- `image`: 图片相对路径
- `conversations`: 对话列表，包含human和gpt的交互

### 对话格式
```json
{
  "image": "my_dataset/screenshot.png",
  "conversations": [
    {
      "from": "human", 
      "value": "<image>\n请点击登录按钮"
    },
    {
      "from": "gpt", 
      "value": "我会帮您点击登录按钮。<click>320, 450</click>"
    }
  ]
}
```

### 支持的动作类型
- `<click>x, y</click>` - 点击坐标(x, y)
- `<type>text</type>` - 输入文本
- `<scroll>x, y</scroll>` - 滚动到坐标(x, y)
- `<drag>x1, y1, x2, y2</drag>` - 从(x1, y1)拖拽到(x2, y2)

## 数据准备步骤

1. 将你的截图放入 `my_dataset/` 目录
2. 根据上述格式编辑 `metadata.jsonl` 文件
3. 确保图片路径和文件名匹配
4. 运行训练脚本开始微调

## 注意事项

- 图片格式支持: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- 坐标系统: 左上角为(0,0)，向右向下递增
- 建议图片分辨率不超过1920x1080以节省显存
- 每个对话应该包含明确的动作指令和对应的坐标
