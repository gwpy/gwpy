# Contributing to GWpy

## Reporting Issues

When opening an issue to report a problem, please try to provide a minimal code
example that reproduces the issue along with details of the operating
system and the Python, NumPy, Astropy, and GWpy versions you are using.

## Contributing Code

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
[Adrienne Lowe](https://github.com/adriennefriend) for a
[PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted for
GWpy based on its use in the [Astropy](https://github.com/astropy/astropy/)
contributing guide.

## Development model

All contributions to GWpy code are made using the
fork and [merge request][mergerequests] [workflow][forkworkflow],
which must then be reviewed by one of the project maintainers.

The basic idea is that all changes are proposed using a dedicated _feature_
branch:

-   Create the fork (if needed) by clicking _Fork_ in the upper-right corner of
  <https://gitlab.com/gwpy/gwpy/> - this only needs to be done once, ever

-   Clone the main repo, calling it `upstream` in the git configuration:

    ```bash
    git clone git@gitlab.com:gwpy/gwpy.git gwpy --origin upstream
    cd gwpy
    ```

-   Add your fork as the `origin` remote (replace `<username>` with your
    GitLab username):

    ```bash
    git remote add origin git@gitlab.com:<username>/gwpy.git
    ```

-   Create a new branch on which to work

    ```bash
    git checkout -b my-new-branch
    ```

-   Make commits to that branch

-   Push changes to your remote on gitlab.com

    ```bash
    git push -u origin my-new-branch
    ```

    This will trigger a small
    [CI/CD pipeline](https://about.gitlab.com/topics/ci-cd/) pipeline
    that will build the project and run the highest-level tests.
    If this pipeline fails, you should modify your code until it passes;
    each `git push` will trigger a new pipeline.

-   Open a merge request on gitlab.com, this will trigger the full CI/CD
    pipeline, including the full test suite and code quality checks, and will
    also trigger code review by one of the GWpy maintainers, who may suggest
    modifications to your proposed changes.

    If you are confident that your changes only impact a particular subsection
    of the project, you can skip jobs in the CI/CD pipeline by adding one or
    more of the following tags _in the merge request description_:

    - `[skip compat]` - skip dependency-compatibility jobs
    - `[skip linux]` - skip Linux-based test jobs (Python and Conda)
    - `[skip macos]` - skip macOS-based test jobs (Python and Conda)
    - `[skip windows]` - skip Windows-based test jobs (Python and Conda)
    - `[skip deploy]` - skip deploying new distributions to PyPI

    The merge request reviewer may ask you to remove one or more of these tags
    before approval, to ensure that nothing breaks.

-   To update your feature branch with the latest changes from the `main` branch
    of the `upstream` repository:

    ```bash
    git pull --rebase upstream main
    ```

    This `rebase` will pull in new changes from the `upstream/main` branch, but will
    restage your commits on top of the newest upstream changes.
    This changes the 'history' of your branch, which means you will need to `force`
    git to push your changes to gitlab.com:

    ```bash
    git push origin my-new-branch --force
    ```

    You can also perform this operation directly from the GitLab merge request page,
    by posting a comment with the following on a new line:

    ```bash
    /rebase
    ```

    For more details, please see <https://docs.gitlab.com/ee/topics/git/git_rebase.html>.

-   Finally, if your Merge Request is merged, you should 'delete the source branch'
    (there's a button), to keep your fork clean.

## Coding guidelines

### Python compatibility

**GWpy code must be compatible with Python >= 3.11.**

### Style

This package follows [PEP 8](https://www.python.org/dev/peps/pep-0008/),
and all code should adhere to that as far as is reasonable.

The first stage in the automated testing of merge requests is a job that runs
the [Ruff](https://docs.astral.sh/ruff/) linter, which checks the style of code
in the repo. You can run this locally before committing changes via:

```bash
python -m ruff .
```

### Testing

GWpy has a fairly complete test suite, covering over 90% of the codebase.
All code contributions should be accompanied by (unit) tests to be executed with
[`pytest`](https://docs.pytest.org/en/latest/), and should cover
all new or modified lines.

You can run the test suite locally from the root of the repository via:

```bash
python3 -m pip install . --group dev
python3 -m pytest gwpy/ examples/
```
