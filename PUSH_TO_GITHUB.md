# 上传到 GitHub

将本目录作为独立仓库推送到 https://github.com/ZeyuLing/PRISM

```bash
cd /path/to/versatilemotion/opensource/prism

# 初始化新仓库（若已有 git 则先 rm -rf .git）
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:ZeyuLing/PRISM.git
git push -u origin main
```

**注意**：`pretrained_models/prism_1.4b/` 下的权重文件已通过 `.gitignore` 排除，不会上传。用户需按 README 从 Hugging Face 下载。
