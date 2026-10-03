#!/usr/bin/env bash
# Refuse any commit that stages a PDF. git has no way to track a file while
# suppressing its diff -- .gitignore applies only to untracked paths -- so the
# only place to enforce "a render enters history deliberately" is the commit
# itself. pre-commit invokes this with the staged PDFs as arguments, and skips
# the hook entirely when none matched.
set -euo pipefail

printf 'Refusing to commit PDF files:\n\n'
printf '  %s\n' "$@"
cat <<'EOF'

A PDF is an opaque binary: git diff reports only that the byte count moved, so
nothing in review can tell an intended render from a wrong one. Commit it on its
own, once you have confirmed it is the render you want:

  SKIP=block-pdf-commit git commit -m "docs(reports): re-render lab N"

EOF
exit 1
