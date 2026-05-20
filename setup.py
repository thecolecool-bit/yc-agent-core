from setuptools import setup, find_packages

setup(
    name='yc-agent',  # 库的名称
    version='1.0.0',  # 版本号
    packages=find_packages(),  # 自动寻找所有的包
    install_requires=['openai', 'fastmcp', 'volcengine-python-sdk','requests'],  # 依赖的第三方库
    author='雨辰',
    description='一款Agent框架',
    python_requires='>=3.6',
)
