#!/bin/bash
# Bilibili Summarizer V2 - 浏览器模式自动脚本
# 自动启动浏览器总结

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 激活虚拟环境 (确保使用隔离环境)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "⚠ Virtual environment not found. Trying global python3..."
fi

echo "====================================="
echo "   Bilibili Summarizer V2 (Browser Mode)"
echo "   确保您已登录 Chrome (首次运行需登录)"
echo "====================================="
echo ""

# Step 1: Fetch
echo "📥 Step 1: 获取稍后再看列表..."
python3 main.py fetch
echo ""

# Step 2: Summarize (Browser Mode)
echo "🤖 Step 2: 启动浏览器生成总结 (自动与 Gemini 交互)..."
# 注意：已开启 --headless 模式，浏览器将在后台运行，不会干扰您的正常使用
python3 main.py summarize --mode browser --max-items 20 --headless
echo ""

# Step 3: EPUB
echo "📚 Step 3: 转换为 EPUB..."
python3 main.py epub
echo ""

# Step 4: Upload to WeChat Reading
echo "📤 Step 4: 上传到微信读书..."
# 默认开启 --headless 模式在后台上传
python3 main.py upload --max-items 20 --headless
echo ""

echo "====================================="
echo "✅ 全部完成！"
echo "EPUB 文件在: output/epub/"
echo "书籍也已在后台尝试上传至微信读书。"
echo "====================================="

# 打开 EPUB 文件夹
open output/epub/
