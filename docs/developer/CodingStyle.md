# Underworld3 House Style

Developers, we welcome contributions to Underworld3 and have put together this style guide to help you
adhere to the patterns of coding and documentation that we have adopted and which we find makes the experience
for users much more predictable.

## Python formatting

The Underworld team adopted the `black` formatting tool and this eliminates most of the need to provide a house style. We think multi-line function definitions (trailing comma after the last argument) work better for clear coding and better version control. This is what we mean:

```python

def rank2_to_voigt(
    v_ij,
    dim,
    covariant=True,
):
    r"""Convert rank 2 tensor ($v_{ij}$) to Voigt (vector) form ($V_I$)"""
```

We do not enforce this formatting at the `git commit` level, but it is strongly encouraged that you set up your code editor to adopt the `black` style. 

## Variable / function names

Underworld3 leans heavily on the `petsc4py` and `sympy` packages. `petsc4py` wraps the `C` / `Fortran` layers of the `PETSc` ecosystem and tends to inherit CamelCase/camelCase variable naming everywhere whereas `sympy`, like other python packages, *favours* snake_case for variables and CamelCase for classes. 

In Underworld3, we prefer to use snake_case naming for functions, properties and variables, camelCase for classes. We also know that we are not completely consistent at present. Sorry !

## Type safety / variable-type hints

We use `typing` for this.

*[To Do: Add instructions and explain how to avoid circular import problems when using uw3 types]*

## Cython considerations

Many `underworld3` objects carry underlying `C` objects that they manage and there is usually some `cython` code that handles the interaction between the two. We use `cython` explicitly for some of this wrapper code and it is present implictly because we make heavy use of the `petsc4py` module. 

`cython` objects may require explicit deletion (because python cannot always drill down to the `C` layer to find objects that are ready for automatic deletion). We strongly encourage the use of the `destroy()` method that is available for many `petsc4py` objects when you are sure the object is no longer required by you. Be aware that lists of pointers to `PETSc` objects may prevent them being automatically deleted and use the `weakref` module to avoid this problem.

Where possible, keep pure python functions in separate source files from `cython` functions as this improves our ability to run the `pdoc` automatic documentation correctly. Generally speaking, the `cython` level of the API is the furthest from the end-user and the least likely to need continual updating. Let's work to keep this code concise, precise, and rarely changed.

## API-level Documentation

The best way to develop `Underworld3` python programs is within the [`jupyter`](jupyter.org) notebook environment. The documentation of the `Underworld3` API assumes that rendered markdown formatting will be available for code highlighting and mathematical equations when the user asks (interactively) for help. 

We use `pdoc` to produce API documentation from the python source code. The `pdoc` configuration assumes all docstrings are markdown and may include mathematics and code snippets. `pdoc` automatically documents arguments to functions and classes, so we do not require these to be described unless there is a need for clarification. 

In the `jupyter` environment `Underworld3` objects display `help` documentation and their internal state through their `self.view()` methods. Many classes have equivalent documentation available before they have been instantiated so a notebook user can progressively construct a model.

**Note**: At the time of writing, the jupyter markdown renderer does not display equations correctly if they are mixed with highlighted code. We often omit code examples when detailed mathematical explanation is required.

## Version Control

We follow the Gitflow Workflow:
https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow/

Features branches should be created for all changes. Once ready for
publishing to the development branch, a pull request should be created and another
team member nominated to review the changes. Pull request should only be merged
once the following criteria are met:

   1. Any new functionality is sufficiently tested
   2. Any new functionality is properly documented
   3. All tests pass
   4. Blog post update for significant changes
   5. Update CHANGES.md
   
## Version Numbering

Underworld follows PEP440 for versioning:

`X.Y.Z`

where

X = Major version number. This will be '3' for the current phase of the design. In Underworld, Major version
    number changes indicate a complete break in the API. The Mathematical formulation and the scripting patterns are continued across the Major versions, but scripts are not backward compatible.

Y = Minor version number. This will increment major feature releases, or with scheduled
    releases (such as quarterly releases).  Unlike SemVer, changes to interface
    may occur with minor version increments.

Z = Micro/Patch version. Backwards compatible bug fixes.

The version number may also be appended with pre-release type designations, for
example 3.0.0b.

Development software will be have the 'dev' suffix, e.g. 3.0.0-dev, and should 
represent the version the development is working towards, so 3.1.0-dev is working
towards release 3.1.0. 

## Testing

The 'test.sh' script  can be used to execute the pytest framework. 
Individual tests are found in the `tests` directory. These are small collections
of unit tests that address parcels of functionality in the code. Tests need to be 
small and fast, and should be orthogonal to other tests as far as possible. 
In the testing framework, we test the deeper levels of the code first (e.g. mesh building)
so that we can use those features for subsequent tests.

We strongly encourage testing to be fine grained with as little as possible additional code that
can go wrong. Please do not just paste in your example notebooks in the hope that `pytest` will 
sort things out. It won't. Somebody else will need to figure out why the test broke sometime in the
future and we don't want that somebody to discover that the problem is not in the unit testing part of the code.
