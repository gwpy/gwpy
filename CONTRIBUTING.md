# Contributing to GWpy

This is [@duncanmmacleod](//github.com/duncanmmacleod/)'s workflow, which might work well for others, it is just a verbose version of the [GitHub flow](https://guides.github.com/introduction/flow/).
The basic idea is to use the `master` branch of your fork as a way of updating your fork with other people's changes that have been merged into the main repo, and then  working on a dedicated _feature branch_ for each piece of work:

- create the fork (if needed) by clicking _Fork_ in the upper-right corner of https://github.com/gwpy/gwpy/ - this only needs to be done once, ever
- clone the fork into a new folder dedicated for this piece of work (replace `<username>` with yout GitHub username):

  ```bash
  git clone https://github.com/<username>/gwpy.git gwpy-my-work  # change gwpy-my-work as appropriate
  cd gwpy-my-work
  ```
  
- link the fork to the upstream 'main' repo:

  ```bash
  git remote add upstream https://github.com/gwpy/gwpy.git
  ```
  
- pull changes from the upstream 'main' repo onto your fork's master branch to pick up other people's changes, then push to your remote to update your fork on github.com

  ```bash
  git pull --rebase upstream master
  git push
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

- open a merge request on github.com
- when the request is merged, you should 'delete the source branch' (there's a button), then just delete the clone of your fork and forget about it

 ```bash
  cd ../
  rm -rf ./gwpy-my-work
  ```
