# `neurogym` developer documentation

If you're looking for user documentation, go [here](README.md).

## IDE settings

We use [Visual Studio Code (VS Code)](https://code.visualstudio.com/) as code editor, which we have set up with some
default [settings](.vscode/settings.json) for formatting. We recommend developers to use VS code with the [recommended extensions](.vscode/extensions.json) to automatically format the code upon saving.

See [VS Code's settings guide](https://code.visualstudio.com/docs/getstarted/settings) for more info.

If using a different IDE, we recommend creating a similar settings file. Feel free to recommend adding this to package
using via a [pull request](#development-conventions).

## Package setup

See installation instructions of the main [README](README.md#installation), but replace the last command by

```bash
pip install -e .'[rl,dev]'
```

Note: you can also run this command after completing the "normal" installation instructions from the [README](README.md#installation).

## Running the tests

You can check that all components were installed correctly, by running [pytest](https://docs.pytest.org/en/stable/#) from your terminal:

```bash
pytest -v
```

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine
how much of the package's code is actually executed during tests. To see the
[coverage](https://pytest-cov.readthedocs.io/en/latest/) results in your terminal, run the following in an activated
conda environment with the dev tools installed:

```bash
coverage run -m pytest
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Style conventions

### Automated style adherence (for VS code users)

If you are using VS code, we recommend installing the [recommended extensions](.vscode/extensions.json): in the Extensions tab, type
`@recommended` and then in the "Workspace Recommendations" list, click the install button. Make sure these extensions
are activated.

This way, the IDE will flag and/or auto-fix upon save most violations to our conventions.

### Linting and formatting (python files)

We use [ruff](https://docs.astral.sh/ruff/) for linting, sorting imports and formatting of python (notebook) files. The
configurations of `ruff` are set [here](.ruff.toml).

Please ensure both linting (`ruff check .`) and formatting (`ruff format .`) rules are adhered to before requesting a review.

### Docstrings

Docstrings should be formatted according to the [google docstring
convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

These conventions are also checked by our linter: `ruff check .`

### Static typing

We use [inline type annotation](https://typing.readthedocs.io/en/latest/source/libraries.html#how-to-provide-type-annotations) for static typing rather than stub files (i.e. `.pyi` files).

Since Python 3.11 is used as dev environment and NeuroGym must support Python version ≥3.10, you may see various typing issues at runtime. Here is [a guide to solve the potential runtime issues](https://mypy.readthedocs.io/en/stable/runtime_troubles.html).

We use [Mypy](http://mypy-lang.org/) as static type checker and the configurations are set in [pyproject.toml](pyproject.toml) file.

For more info about static typing and mypy, see:

- [Static typing with Python](https://typing.readthedocs.io/en/latest/index.html#)
- [Mypy doc](https://mypy.readthedocs.io/en/stable/)

You can run `mypy .` to check for static typing violations before requesting a review.

### Formatting non-python files

We use [prettier](https://prettier.io/) for formatting most non-python files.

You can run `npx prettier --check .` to check the formatting of, or `npx prettier --write .` to auto-format non-python files.

## User Documentation

We use [MkDocs](https://www.mkdocs.org/) and its theme [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) to generate documentations. The configurations of MkDocs are set in [mkdocs.yml](mkdocs.yml) file.

To watch the changes of current doc in real time, run:

```shell
mkdocs serve
```

or to watch src and docs directories

```shell
mkdocs serve -w docs -w src
```

Then open your browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### Publishing the docs

The docs are published on GitHub pages. We use [mike](https://github.com/jimporter/mike) to deploy the docs to the `gh-pages` branch and to manage the versions of docs.

Docs are automatically deployed upon each new release to PyPi as part of our [PyPi
release](.github/workflows/release_pypi.yml) workflow and named identically to the newest GitHub version number (i.e. `vX.Y.Z`).

To manually deploy (additional) versions of the docs, run:

```shell
mike deploy -p -u <docs_version_name>
```

Add `latest` to the end of the command above, to make this the latest default version of the docs (i.e. the landing page
of https://neurogym.github.io/).

To remove a given version of the documentation, run

```shell
mike delete <docs_version_name>
```

Note that all mike operations above will be recorded as git commits to the `gh-pages` branch.

Use `mike serve` to review all versions of the site that have been committed to the `gh-pages` branch. It’s part of the `mike` tool, which manages versioned documentation for `mkdocs`. This command is best for verifying the production website as it exists in the repository.

If you have uncommitted changes that you want to preview, use `mkdocs serve` instead of `mike serve`. This will allow you to view your changes locally without committing them first.

## Versioning

We adhere to [semantic versioning](https://semver.org/) standards. In brief this means using `X.Y.Z` versioning, where

- X = `major` version: representing API-incompatible changes from the previous version
- Y = `minor` version: representing added functionality that is backwards compatible to previous versions
- Z = `patch` version: representing backward compatible bug fixes were made of previous version

Bumping the version consistently is done using [bump-my-version](https://callowayproject.github.io/bump-my-version/),
which automatically updates all mentions of the current version throughout the package, as defined in the tool's
[settings](.bumpversion.toml). Use `major`, `minor`, or `patch` as the version level in the following command to update
the version:

```bash
bump-my-version bump <version level>
```

## Branching workflow

We use a [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)-inspired branching workflow for development. DeepRank2's repository is based on two main branches with infinite lifetime:

- `main` — this branch contains production (stable) code. All development code is merged into `main` in sometime.
- `dev` — this branch contains pre-production code. When the features are finished then they are merged into `dev`.

During the development cycle, three main supporting branches are used:

- Feature branches - Branches that branch off from `dev` and must merge into `dev`: used to develop new features for the upcoming releases.
- Hotfix branches - Branches that branch off from `main` and must merge into `main` and `dev`: necessary to act immediately upon an undesired status of `main`.
- Release branches - Branches that branch off from `dev` and must merge into `main` and `dev`: support preparation of a new production release. They allow many minor bug to be fixed and preparation of meta-data for a release.

### Committing (changes to) Jupyter notebooks

Please commit any (changes to) Jupyter notebooks as empty notebooks to improve the reviewability. Do this by:

1. clearing all outputs to remove from the notebook, and
2. restarting the kernel ("restart" button) to reset which will the `execution_count` of all cells.

Jupyter notebooks are notoriously difficult to review effectively. In this way, it is kept at least somewhat manageable.

## Development conventions

We highly appreciate external contributions to our package! We do request developers to adhere to the following conventions:

- Issues
  - Before working on any kind of development, please check existing issues to see whether there has been any
    discussion and/or ongoing work on this topic.
  - If no issue exists, please open one to allow discuss whether the development is desirable to the maintainers.
- Branching
  - Always branch from `dev` branch, unless there is the need to fix an undesired status of `main`. See above for more details about the branching workflow adopted.
  - Our branch naming convention is: `<issue_number>_<description>_<author_name>`.
- Pull Requests
  - New developments must proceed via a pull request (PR) before they can be merged to either `main` or `dev` branches.
  - When creating a pull request, please use the following naming convention: `<type>: <description>`. Example _types_
    are `fix:`, `feat:`, `docs:`, and others based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).

## Making a release

### Automated release workflow

1. **IMP0RTANT:** Create a PR for the release branch, targeting the `main` branch. Ensure there are no conflicts and that all checks pass successfully. Release branches are typically: traditional [release branches](https://nvie.com/posts/a-successful-git-branching-model/#release-branches) (these are created from the `dev` branch), or [hotfix branches](https://nvie.com/posts/a-successful-git-branching-model/#hotfix-branches) (these are created directly from the `main` branch).
   - if everything goes well, this PR will automatically be closed after the draft release is created.
2. Navigate to [Draft Github Release](https://github.com/neurogym/neurogym/actions/workflows/release_github.yml) on the [Actions](https://github.com/neurogym/neurogym/actions) tab.
3. On the right hand side, you can select the [version level update](#versioning) ("patch", "minor", or "major") and which branch to release from.
   - [Follow semantic versioning conventions](https://semver.org/)
   - Note that you cannot release from `main` (the default shown) using the automated workflow. To release from `main`
     directly, you must [create the release manually](#manually-create-a-release).
4. Visit [Actions](https://github.com/neurogym/neurogym/actions) tab to check whether everything went as expected.
   - NOTE: there are two separate jobs in the workflow: "draft_release" and "tidy_workspace". The first creates the draft release on github, while the second merges changes into `dev` and closes the PR.
     - If "draft_release" fails, then there are likely merge conflicts with `main` that need to be resolved first. No release draft is created and the "tidy_workspace" job does not run. Coversely, if this action is succesfull, then the release branch (including a version bump) have been merged into the remote `main` branch.
     - If "draft_release" is succesfull but "tidy_workspace" fails, then there are likely merge conflicts with `dev` that are not conflicts with `main`. In this case, the draft release is created (and changes were merged into the remote `main`). Conflicts with `dev` need to be resolved with `dev` by the user.
     - If both jobs succeed, then the draft release is created and the changes are merged into both remote `main` and `dev` without any problems and the associated PR is closed. Also, the release branch is deleted from the remote repository.
5. Navigate to the [Releases](https://github.com/neurogym/neurogym/releases) tab and click on the newest draft
   release that was just generated.
6. Click on the edit (pencil) icon on the right side of the draft release.
7. Check/adapt the release notes and make sure that everything is as expected.
8. Check that "Set as the latest release is checked".
9. Click green "Publish Release" button to convert the draft to a published release on GitHub.
   - This will automatically trigger [another GitHub workflow](https://github.com/neurogym/neurogym/actions/workflows/release_pypi.yml) that will take care of publishing the package on PyPi.

#### Updating the token

In order for the workflow above to be able to bypass the branch protection on `main` and `dev`, a token with admin priviliges for the current repo is required. Below are instructions on how to create such a token.
NOTE: the current token (associated to @DaniBodor) allowing to bypass branch protection will expire on 9 July 2025. To update the token do the following:

1. [Create a personal access token](https://github.com/settings/tokens/new) from a GitHub user account with admin
   priviliges for this repo.
2. Check all the "repo" boxes and the "workflow" box, set an expiration date, and give the token a note.
3. Click green "Generate token" button on the bottom
4. Copy the token immediately, as it will not be visible again later.
5. Navigate to the [secrets settings](https://github.com/neurogym/neurogym/settings/secrets/actions).
   - Note that you need admin priviliges to the current repo to access these settings.
6. Edit the `GH_RELEASE` key giving your access token as the new value.

### Manually create a release

0. Make sure you have all required developers tools installed `pip install -e .'[test]'`.
1. Create a `release-` branch from `main` (if there has been an hotfix) or `dev` (regular new production release).
2. Prepare the branch for the release (e.g., removing the unnecessary dev files, fix minor bugs if necessary). Do this by ensuring all tests pass `pytest -v` and that linting (`ruff check`) and formatting (`ruff format --check`) conventions are adhered to.
3. Decide on the [version level increase](#versioning), following [semantic versioning
   conventions](https://semver.org/). Use [bump-my-version](https://github.com/callowayproject/bump-my-version):
   `bump-my-version bump <level> --commit --tag -vv` to update the version throughout the package.
4. Merge the release branch into `main` and `dev`.
5. On the [Releases page](https://github.com/neurogym/neurogym/releases):
   1. Click "Draft a new release"
   2. By convention, use `v<version number>` as both the release title and as a tag for the release.
   3. Click "Generate release notes" to automatically load release notes from merged PRs since the last release.
   4. Adjust the notes as required.
   5. Ensure that "Set as latest release" is checked and that both other boxes are unchecked.
   6. Hit "Publish release".
      - This will automatically trigger a [GitHub
        workflow](https://github.com/neurogym/neurogym/actions/workflows/release.yml) that will take care of publishing
        the package on PyPi.
