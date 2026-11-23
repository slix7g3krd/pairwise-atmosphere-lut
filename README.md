# pairwise-atmosphere-lut

## 领域

游戏、图形与多媒体

## 主题

大气散射 LUT

## 说明

海拔重算、边界插值与 LUT 栅栏。

初始状态下 `graphics/atmosphere_lut.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
