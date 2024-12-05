# `deeprank2` developer documentation

If you're looking for user documentation, go [here](README.md).

## Code editor

We use [Visual Studio Code (VS Code)](https://code.visualstudio.com/) as code editor.
The VS Code settings for this project can be found in [.vscode](.vscode).
The settings will be automatically loaded and applied when you open the project with VS Code.
See [the guide](https://code.visualstudio.com/docs/getstarted/settings) for more info about workspace settings of VS Code.

## Package setup

After having followed the [installation instructions](https://github.com/DeepRank/deeprank2#installation) and installed all the dependencies of the package, the repository can be cloned and its editable version can be installed:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

## Running the tests

You can check that all components were installed correctly, using pytest.
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

Run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

## Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests. In an activated conda environment with the development tools installed, inside the package directory, run:

```bash
coverage run -m pytest
```

This runs tests and stores the result in a `.coverage` file. To see the results on the command line, run:

```bash
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting, sorting imports and formatting of python (notebook) files. The configurations of `ruff` are set in [pyproject.toml](pyproject.toml) file.

If you are using VS code, please install and activate the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to automatically format and check linting.

Otherwise, please ensure check both linting (`ruff check .`) and formatting (`ruff format .`) before requesting a review.

We use [prettier](https://prettier.io/) for formatting most other files. If you are editing or adding non-python files and using VS code, the [Prettier extension](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) can be installed to auto-format these files as well.

## Versioning

Bumping the version across all files is done before creating a new package release, running `bump2version [part]` from command line after having installed [bump2version](https://pypi.org/project/bump2version/) on your local environment. Instead of `[part]`, type the part of the version to increase, e.g. minor. The settings in `.bumpversion.cfg` will take care of updating all the files containing version strings.

## Branching workflow

We use a [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)-inspired branching workflow for development. DeepRank2's repository is based on two main branches with infinite lifetime:

- `main` — this branch contains production (stable) code. All development code is merged into `main` in sometime.
- `dev` — this branch contains pre-production code. When the features are finished then they are merged into `dev`.

During the development cycle, three main supporting branches are used:

- Feature branches - Branches that branch off from `dev` and must merge into `dev`: used to develop new features for the upcoming releases.
- Hotfix branches - Branches that branch off from `main` and must merge into `main` and `dev`: necessary to act immediately upon an undesired status of `main`.
- Release branches - Branches that branch off from `dev` and must merge into `main` and `dev`: support preparation of a new production release. They allow many minor bug to be fixed and preparation of meta-data for a release.

### Development conventions

- Branching
  - When creating a new branch, please use the following convention: `<issue_number>_<description>_<author_name>`.
  - Always branch from `dev` branch, unless there is the need to fix an undesired status of `main`. See above for more details about the branching workflow adopted.
- Pull Requests
  - When creating a pull request, please use the following convention: `<type>: <description>`. Example _types_ are `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).

## Making a release

### Automated release workflow:

1. **IMP0RTANT:** Create a PR for the release branch, targeting the `main` branch. Ensure there are no conflicts and that all checks pass successfully. Release branches are typically: traditional [release branches](https://nvie.com/posts/a-successful-git-branching-model/#release-branches) (these are created from the `dev` branch), or [hotfix branches](https://nvie.com/posts/a-successful-git-branching-model/#hotfix-branches) (these are created directly from the `main` branch).
   - if everything goes well, this PR will automatically be closed after the draft release is created.
2. Navigate to [Draft Github Release](https://github.com/DeepRank/deeprank2/actions/workflows/release_github.yml)
   on the [Actions](https://github.com/DeepRank/deeprank2/actions) tab.
3. On the right hand side, you can select the level increase ("patch", "minor", or "major") and which branch to release from.
   - [Follow semantic versioning conventions](https://semver.org/) to chose the level increase:
     - `patch`: when backward compatible bug fixes were made
     - `minor`: when functionality was added in a backward compatible manner
     - `major`: when API-incompatible changes have been made
   - Note that you cannot release from `main` (the default shown) using the automated workflow. To release from `main`
     directly, you must [create the release manually](#manually-create-a-release).
4. Visit [Actions](https://github.com/DeepRank/deeprank2/actions) tab to check whether everything went as expected.
   - NOTE: there are two separate jobs in the workflow: "draft_release" and "tidy_workspace". The first creates the draft release on github, while the second merges changes into `dev` and closes the PR.
     - If "draft_release" fails, then there are likely merge conflicts with `main` that need to be resolved first. No release draft is created and the "tidy_workspace" job does not run. Coversely, if this action is succesfull, then the release branch (including a version bump) have been merged into the remote `main` branch.
     - If "draft_release" is succesfull but "tidy_workspace" fails, then there are likely merge conflicts with `dev` that are not conflicts with `main`. In this case, the draft release is created (and changes were merged into the remote `main`). Conflicts with `dev` need to be resolved with `dev` by the user.
     - If both jobs succeed, then the draft release is created and the changes are merged into both remote `main` and `dev` without any problems and the associated PR is closed. Also, the release branch is deleted from the remote repository.
5. Navigate to the [Releases](https://github.com/DeepRank/deeprank2/releases) tab and click on the newest draft
   release that was just generated.
6. Click on the edit (pencil) icon on the right side of the draft release.
7. Check/adapt the release notes and make sure that everything is as expected.
8. Check that "Set as the latest release is checked".
9. Click green "Publish Release" button to convert the draft to a published release on GitHub.
   - This will automatically trigger [another GitHub workflow](https://github.com/DeepRank/deeprank2/actions/workflows/release_pypi.yml) that will take care of publishing the package on PyPi.

#### Updating the token:

In order for the workflow above to be able to bypass the branch protection on `main` and `dev`, a token with admin priviliges for the current repo is required. Below are instructions on how to create such a token.
NOTE: the current token (associated to @DaniBodor) allowing to bypass branch protection will expire on 9 July 2025. To update the token do the following:

1. [Create a personal access token](https://github.com/settings/tokens/new) from a GitHub user account with admin
   priviliges for this repo.
2. Check all the "repo" boxes and the "workflow" box, set an expiration date, and give the token a note.
3. Click green "Generate token" button on the bottom
4. Copy the token immediately, as it will not be visible again later.
5. Navigate to the [secrets settings](https://github.com/DeepRank/deeprank2/settings/secrets/actions).
6. Edit the `GH_RELEASE` key giving your access token as the new value.

### Manually create a release:

0. Make sure you have all required developers tools installed `pip install -e .'[test]'`.
1. Create a `release-` branch from `main` (if there has been an hotfix) or `dev` (regular new production release).
2. Prepare the branch for the release (e.g., removing the unnecessary dev files, fix minor bugs if necessary). Do this by ensuring all tests pass `pytest -v` and that linting (`ruff check`) and formatting (`ruff format --check`) conventions are adhered to.
3. Bump the version using [bump-my-version](https://github.com/callowayproject/bump-my-version): `bump-my-version bump <level>`
   where level must be one of the following ([following semantic versioning conventions](https://semver.org/)):
   - `major`: when API-incompatible changes have been made
   - `minor`: when functionality was added in a backward compatible manner
   - `patch`: when backward compatible bug fixes were made
4. Merge the release branch into `main` and `dev`.
5. On the [Releases page](https://github.com/DeepRank/deeprank2/releases):
   1. Click "Draft a new release"
   2. By convention, use `v<version number>` as both the release title and as a tag for the release.
   3. Click "Generate release notes" to automatically load release notes from merged PRs since the last release.
   4. Adjust the notes as required.
   5. Ensure that "Set as latest release" is checked and that both other boxes are unchecked.
   6. Hit "Publish release".
      - This will automatically trigger a [GitHub
        workflow](https://github.com/DeepRank/deeprank2/actions/workflows/release.yml) that will take care of publishing
        the package on PyPi.

## UML

Code-base class diagrams updated on 02/11/2023, generated with https://www.gituml.com (save the images and open them in the browser for zooming).

- Data processing classes and functions: <img src="./tests/utils/uml_data_processing.svg" width="50">
- ML pipeline classes and functions: <img src="./tests/utils/uml_training.svg" width="50">
