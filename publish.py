#!/usr/bin/env python3
"""
发布脚本：自动化 causal-sklearn 包到 PyPI 的发布流程

使用方法:
1. 测试发布: python publish.py --test
2. 正式发布: python publish.py --release
3. 仅构建: python publish.py --build-only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """运行命令并检查返回值"""
    print(f"🔄 {description}")
    print(f"   执行: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 错误: {description}")
        print(f"   标准输出: {result.stdout}")
        print(f"   错误输出: {result.stderr}")
        sys.exit(1)
    else:
        print(f"✅ 完成: {description}")
        if result.stdout.strip():
            print(f"   输出: {result.stdout.strip()}")
    
    return result


def clean_build():
    """清理构建目录"""
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for dir_pattern in dirs_to_clean:
        run_command(f"rm -rf {dir_pattern}", f"清理 {dir_pattern}")


def build_package():
    """构建包"""
    print("\n📦 开始构建包...")
    
    # 清理旧的构建文件
    clean_build()
    
    # 安装构建依赖
    run_command("pip install --upgrade build twine", "安装构建工具")
    
    # 构建包
    run_command("python -m build --no-isolation", "构建源码和wheel包 (使用 --no-isolation)")
    
    # 检查包
    run_command("twine check dist/*", "检查包完整性")


def upload_to_test_pypi():
    """上传到 TestPyPI"""
    print("\n🧪 上传到 TestPyPI...")
    
    print("请确保您已经:")
    print("1. 在 https://test.pypi.org/ 注册账户")
    print("2. 设置了 API token")
    print("3. 配置了 ~/.pypirc 或准备手动输入凭据")
    
    confirm = input("\n是否继续上传到 TestPyPI? (y/N): ")
    if confirm.lower() != 'y':
        print("取消上传到 TestPyPI")
        return
    
    run_command("twine upload --repository testpypi dist/*", "上传到 TestPyPI")
    
    print("\n✅ 上传到 TestPyPI 完成!")
    print("🔗 查看您的包: https://test.pypi.org/project/causal-sklearn/")
    print("📦 测试安装: pip install --index-url https://test.pypi.org/simple/ causal-sklearn")


def upload_to_pypi():
    """上传到正式 PyPI"""
    print("\n🚀 上传到正式 PyPI...")
    
    print("⚠️  警告: 这将发布到正式的 PyPI!")
    print("请确保您已经:")
    print("1. 在 https://pypi.org/ 注册账户")
    print("2. 设置了 API token")
    print("3. 在 TestPyPI 上测试过包")
    print("4. 更新了版本号")
    
    confirm = input("\n确认发布到正式 PyPI? (y/N): ")
    if confirm.lower() != 'y':
        print("取消发布到 PyPI")
        return
        
    double_confirm = input("再次确认发布到正式 PyPI? 这不可撤销! (y/N): ")
    if double_confirm.lower() != 'y':
        print("取消发布到 PyPI")
        return
    
    run_command("twine upload dist/*", "上传到正式 PyPI")
    
    print("\n🎉 发布到 PyPI 成功!")
    print("🔗 查看您的包: https://pypi.org/project/causal-sklearn/")
    print("📦 现在用户可以使用: pip install causal-sklearn")


def test_local_install():
    """测试本地安装"""
    print("\n🧪 测试本地安装...")
    
    # 在临时虚拟环境中测试
    print("建议在新的虚拟环境中测试:")
    print("1. python -m venv test_env")
    print("2. source test_env/bin/activate  # Linux/Mac")
    print("3. pip install dist/*.whl")
    print("4. python -c 'import causal_sklearn; print(causal_sklearn.__version__)'")


def main():
    parser = argparse.ArgumentParser(description="Causal-sklearn 发布工具")
    parser.add_argument("--test", action="store_true", help="发布到 TestPyPI")
    parser.add_argument("--release", action="store_true", help="发布到正式 PyPI")
    parser.add_argument("--build-only", action="store_true", help="仅构建包，不上传")
    
    args = parser.parse_args()
    
    # 检查是否在正确的目录
    if not Path("setup.py").exists() or not Path("causal_sklearn").exists():
        print("❌ 错误: 请在 causal-sklearn 项目根目录运行此脚本")
        sys.exit(1)
    
    print("🚀 Causal-sklearn 发布工具")
    print("=" * 50)
    
    # 总是先构建
    build_package()
    
    if args.build_only:
        print("\n✅ 仅构建模式完成")
        test_local_install()
        
    elif args.test:
        upload_to_test_pypi()
        
    elif args.release:
        upload_to_pypi()
        
    else:
        print("\n🤔 请选择操作:")
        print("  --build-only  : 仅构建包")
        print("  --test       : 发布到 TestPyPI")  
        print("  --release    : 发布到正式 PyPI")
        print("\n示例: python publish.py --test")


if __name__ == "__main__":
    main()