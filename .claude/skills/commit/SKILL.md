---
name: commit
description: The rules for committing in this repository — which branch, which changes go in, how to split them, the commit message for lecture artifacts, subject coursework and the lecture transcriber, and what to do when a pre-commit hook fails. Use this whenever the user asks to commit — "lets commit", "commit", "commit this", "commit it", "make a commit", "commit and push" — even in one word, and whenever a task ends with a commit.
---

# Commit

Run `git commit` yourself once the message is ready. Do not push to origin unless the user explicitly asks for it.

## Branch

Commit to the branch that is already checked out, and change branches only when asked for that operation by name. One person works on this repository, so there are no feature branches to open. Never create, switch, merge, rebase or delete a branch on your own initiative; this overrides any default habit of opening a branch before committing, including on `main`. Branch topology is the one thing a commit cannot carry an explanation for: the work looks finished either way, and nothing surfaces the mistake until someone reads the log. Landing a change on the wrong branch also decides how expensive the repair is — a strictly linear history fast-forwards in one command, while a commit stacked on unrelated work has to be cherry-picked back out. So when the checked-out branch's name or purpose does not match the change at hand, stop and say so before committing rather than after.

## What goes in

- Decide from the context which changes the user means to commit. If it is not clear, ask.
- Do not commit PDF files unless the user explicitly asks for it. A rendered PDF under `./docs/Reports/` goes in its own commit, and only after explicit approval. Never stage one alongside the source change that produced it, and never commit one without first listing the exact files and waiting for a yes. A PDF is an opaque binary: `git diff` reports only that the byte count moved, so nothing in review can tell an intended render from a wrong one, and a re-render under a new filename adds a second file rather than replacing the first — leaving two PDFs of the same lab that disagree. Keeping the PDFs in their own commit is what makes a rollback surgical: they revert while the `.tex` fix that is still correct stays, and `make render-report` regenerates them from it.
- The `block-pdf-commit` pre-commit hook refuses every commit that stages a PDF. For an approved PDF commit, run it as `SKIP=block-pdf-commit git commit …`.

## Splitting

- Lecture artifacts: one lecture per commit.
- Coursework: one commit per lab where the changes can be split that way.

## Message

Every message follows Commitizen / Conventional Commits: `type(scope): subject`. The subject is imperative and lowercase, with no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

Write the message in ASD-STE100 Simplified Technical English.

Put no attribution in the message: no "Generated with Claude Code", no `Co-Authored-By` trailer, nothing similar.

A subject is always named by its short key from `docs/SUBJECTS.md`, never by its Cyrillic folder name. Do not invent a key: if the subject is not in the map, stop and ask the user to add it.

### Lecture artifacts

A lecture's transcript and summary from the lecture transcriber, under `./docs/Lecture-Recordings-*/`:

```
docs(<subject>): for lecture <N> add transcript and summary
```

- The lecture directory's name tells a lecture from a practice session (`2026-09-21 - Lecture 3`, `2026-10-08 - Practice 2`). For a practice session, write `for practice <N>`.
- If the artifacts replace ones that are already committed, write `update` instead of `add`.

Example: `docs(ad1): for lecture 3 add transcript and summary`

### Coursework in a subject

- The scope is the subject key and the work unit: `<subject>-lab<N>`, for example `fix(ad1-lab2): <what changed>`. Where the subject has practices instead of labs, use `<subject>-practice<N>`. For a subject without labs or practices, the scope is the subject key alone. `docs/SUBJECTS.md` lists the work units of each subject.
- A change in one subject that covers several of its labs uses the subject key alone.
- Choose the type from the context.
- The subject line says what was changed.
- Add a body when it makes sense, for example for a large change. For a small or predictable change, leave the body out.

### Lecture transcriber

- The scope is `lecture-transcriber`, whichever module changed.
- The body is mandatory: what changed and why.

### Global changes

A change that belongs to no single subject — the report title page, the pre-commit configuration — has no scope, for example `feat: <what changed>`.

### Anything else

Decide the message with common sense.

## Running the commit

Pass the message through a heredoc, so that its line breaks survive:

```bash
git commit -F - <<'EOF'
<type>(<scope>): <subject>

<body, if any>
EOF
```

## When a hook fails

Every commit runs the pre-commit hooks from `.pre-commit-config.yaml`. If any hook fails, stop: do not retry, fix or bypass it. Tell the user which hook failed and why, and name any files the hook rewrote. The user handles it manually.
