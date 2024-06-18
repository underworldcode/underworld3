This page provides information about contributing to Underworld’s codebase.

For contributions to Underworld models please go https://github.com/underworld-community

---- 

We welcome contributions to Underworld’s codebase in the form of:

  * Code improvements or completion of missing functionality
  * Bug reports and bug fixes
  * Suggestions / Requests
  * Documentation modifications (including docstrings)

For Bug reports and Suggestions / Requests please submit an Issue on the Underworld GitHub Issue Tracker. 
Please tag the Issue with a given Label to help us assess the issue and provide simple scripts that explain how to 
reproduce the problem.

Click here to submit an Issue https://github.com/underworldcode/underworld3/issues

For Code / Documentation changes please submit a GitHub Pull Request (PR). This allows us to review and discuss the contributions before merging it into our `development` branch. For creating Pull Request (PR) we recommend following the workflow outlined https://guides.github.com/activities/forking/.
More specifically:

1. Fork Underworld via GitHub and clone it to your machine.

``` bash

    git clone https://github.com/{YOUR_GITHUB_ACCOUNT}/underworld3
```

2. Add the master Underworld repository as an additional remote source (named `uwmaster`) for your local repo and pull down its latest changesets. Checkout to the master/development repo state, and then create a new local branch which will contain your forthcoming changes.

``` bash
  
    git remote add uw3 https://github.com/underworldcode/underworld3
    git pull uw3
    git checkout uw3/development
    git checkout -b newFeature

```
     
3. Make your changes! Remember to write comments, a test if applicable and follow the code style of the project<!-- (see `./docs/development/guidelines.md` for details). NB: this is on the todo list for uw3 -->

4. Push your changes to your GitHub fork and then submit a PR to the `development` branch of Underworld via Github.