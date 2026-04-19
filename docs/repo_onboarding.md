# Repo Onboarding

To add a new upstream repo:

1. create `configs/repos/<name>.yaml`
2. add a repo adapter under `src/clab/repos/` if repo-specific behavior is needed
3. add one or more episodes under `data/episodes/<name>/`
4. define visible docs, hidden evaluation, and scoring artifacts for each episode
5. add targeted tests if the repo requires custom environment behavior

Keep repo-specific logic inside `src/clab/repos/` so the generic benchmark modules
remain reusable.

