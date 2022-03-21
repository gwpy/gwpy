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

Contributions to GWpy are made via pull requests from GitHub users' forks of
the main [gwpy repositories](https://github.com/gwpy/gwpy), following the
[GitHub flow](https://guides.github.com/introduction/flow/) workflow.
The basic idea is that all changes are proposed using a dedicated _feature_
branch:

- create the fork (if needed) by clicking _Fork_ in the upper-right corner of
  <https://github.com/gwpy/gwpy/> - this only needs to be done once, ever

- clone the main repo, calling it `upstream` in the git configuration:

  ```bash
  git clone https://github.com/gwpy/gwpy.git gwpy --origin upstream
  cd gwpy
  ```

- add your fork as the `origin` remote (replace `<username>` with your
  GitHub username):

  ```bash
  git remote add origin git@github.com:<username>/gwpy.git
  ```

- create a new branch on which to work

  ```bash
  git checkout -b my-new-branch
  ```

- make commits to that branch

- push changes to your remote on github.com

  ```bash
  git push -u origin my-new-branch
  ```

- open a merge request on github.com, this will trigger a code review by one
  of the GWpy maintainers, who may suggest modifications to your proposed
  changes

- to update your feature branch with the latest changes from the `main` branch
  of the `upstream` repository:

  ```bash
  git pull --rebase upstream main
  ```

  This `rebase` will pull in new changes from the `upstream/main` branch, but will
  restage your commits on top of the newest upstream changes.
  This changes the 'history' of your branch, which means you will need to `force`
  git to push your changes to github.com:

  ```bash
  git push origin my-new-branch --force
  ```

- finally, if your Pull Request is merged, you should 'delete the source branch'
  (there's a button), to keep your fork clean

## Coding guidelines

### Python compatibility

**GWpy code must be compatible with Python >= 3.7.**

### Style

This package follows [PEP 8](https://www.python.org/dev/peps/pep-0008/),
and all code should adhere to that as far as is reasonable.

The first stage in the automated testing of pull requests is a job that runs
the [`flake8`](http://flake8.pycqa.org) linter, which checks the style of code
in the repo. You can run this locally before committing changes via:

```bash
python3 -m flake8
```

### Testing

GWpy has a fairly complete test suite, covering over 90% of the codebase.
All code contributions should be accompanied by (unit) tests to be executed with
[`pytest`](https://docs.pytest.org/en/latest/), and should cover
all new or modified lines.

You can run the test suite locally from the root of the repository via:

```bash
python3 -m pytest gwpy/
```
