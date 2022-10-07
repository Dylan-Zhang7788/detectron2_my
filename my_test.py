# 专门测试代码用的一个py文件

__all__ = [k for k in globals().keys() if not k.startswith("_")]

print(__all__)