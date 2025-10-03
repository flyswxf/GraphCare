# Git 常用命令速查（Windows/PowerShell 通用）

本文总结了常用 Git 命令，覆盖以下场景：
- 拉取指定分支并与远程同步
- 回退到某一个版本（临时查看、回滚提交、重置分支）
- 获取所有分支用于 IDE 可视化（不修改代码）
- 以及分支管理、提交与暂存、远程配置等常用操作

## 拉取指定分支并与远程同步
- 切换到目标分支（已存在）：
  - `git checkout <branch>`
  - 或 `git switch <branch>`
- 拉取更新（已设置上游）：
  - `git pull`
- 拉取更新（未设置上游）：
  - `git pull origin <branch>`
- 设置上游（便于后续直接 `git pull`）：
  - `git push -u origin <branch>`
  - 或 `git branch --set-upstream-to=origin/<branch>`
- 精确同步到远程（覆盖本地改动，危险操作）：
  - `git fetch origin`
  - `git reset --hard origin/<branch>`
  - 可选清理未跟踪文件：`git clean -fd`

## 回退到某一个版本
- 查看提交历史（简洁图形）：
  - `git log --oneline --decorate --graph`
- 临时查看旧版本（不改动当前分支指向，进入分离头状态）：
  - `git checkout <commit>`
- 将当前分支回退到某次提交（重写分支指向，丢弃本地未提交改动）：
  - `git reset --hard <commit>`
- 只回滚某次提交（保留历史，生成反向提交）：
  - `git revert <commit>`
- 回退后推送到远程（可能需要强推，谨慎）：
  - `git push` 或 `git push -f origin <branch>`

## 获取所有分支用于可视化（不修改代码）
- 获取远程所有更新（不合并、不修改工作区）：
  - `git fetch --all --prune`
- 查看所有分支列表：
  - 本地与远程：`git branch -a`
  - 仅远程：`git branch -r`
- 查看 Git 树（IDE 可视化或命令行图）：
  - `git log --oneline --decorate --graph --all`
  - 若已安装：`gitk --all`（可视化工具）

## 分支管理
- 创建新分支并切换：`git checkout -b <branch>`
- 推送并设置上游：`git push -u origin <branch>`
- 删除本地分支：`git branch -d <branch>`（强制：`-D`）
- 删除远程分支：`git push origin --delete <branch>`

## 提交与暂存
- 查看状态：`git status`
- 查看改动：`git diff`（未暂存）、`git diff --staged`（已暂存）
- 暂存并提交：
  - `git add -A`
  - `git commit -m "message"`
- 修改最近一次提交信息（未推送前）：`git commit --amend`
- 暂存工作区改动：
  - `git stash`
  - 恢复：`git stash pop`

## 安全恢复
- 放弃某文件的未暂存修改：`git restore <file>` 或 `git checkout -- <file>`
- 取消暂存某文件：`git restore --staged <file>`

## 远程配置
- 查看远程：`git remote -v`
- 添加远程：`git remote add origin <url>`
- 修改远程 URL：`git remote set-url origin <url>`

## 常用查看
- 查看某文件历史：`git log -- <file>`
- 查看某次提交详情：`git show <commit>`
- 比较两个提交：`git diff <commitA>..<commitB>`

## 注意事项
- `reset --hard` 会丢弃本地未提交改动，谨慎使用。
- 强推 `push -f` 会重写远程历史，仅在确认需要时使用。
- 切换分支前建议保证工作区干净或使用 `git stash` 暂存。