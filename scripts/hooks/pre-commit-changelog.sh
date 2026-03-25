#!/usr/bin/env bash
# pre-commit-changelog.sh
#
# 检查本次暂存的文件中是否包含生产代码改动（src/ 或 scripts/），
# 若 CHANGELOG.md 本身未被暂存更新，则发出警告提醒开发者手动记录变更。
#
# 安装方式（在仓库根目录执行）：
#   cp scripts/hooks/pre-commit-changelog.sh .git/hooks/pre-commit-changelog
#   chmod +x .git/hooks/pre-commit-changelog
#   # 然后在 .git/hooks/pre-commit 末尾追加一行：
#   #   bash "$(git rev-parse --show-toplevel)/scripts/hooks/pre-commit-changelog.sh"
#
# 或通过 pre-commit 框架管理，详见 docs/workflow/changelog.md。

set -euo pipefail

# 获取本次已暂存文件列表
STAGED=$(git diff --cached --name-only 2>/dev/null || true)

if [ -z "$STAGED" ]; then
  exit 0
fi

# 判断是否有生产代码变更
PROD_CHANGED=$(echo "$STAGED" | grep -E '^(src/|scripts/)' || true)

if [ -z "$PROD_CHANGED" ]; then
  exit 0
fi

# 判断 CHANGELOG.md 是否也被暂存
CHANGELOG_STAGED=$(echo "$STAGED" | grep -E '^CHANGELOG\.md$' || true)

if [ -z "$CHANGELOG_STAGED" ]; then
  echo ""
  echo "⚠️  [changelog] 检测到生产代码变更，但 CHANGELOG.md 未更新。"
  echo ""
  echo "   受影响文件："
  echo "$PROD_CHANGED" | sed 's/^/     - /'
  echo ""
  echo "   请在提交前更新 CHANGELOG.md（在 [Unreleased] 节下追加条目），"
  echo "   或者确认本次变更不需要记录后，运行："
  echo "     git commit --no-verify"
  echo ""
  exit 1
fi

exit 0
