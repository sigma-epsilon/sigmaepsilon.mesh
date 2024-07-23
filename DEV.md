# Developers Guide

## Setting Up a Development Enviroment

If you are a developer, you should use `Poetry` to manage the project. You can install `Poetry` globally using the following command:

```console
pip install poetry
```

Don't worry. This will be the only package that you have to install into your global Python enviroment, as Poetry handles all the rest and isolates your packages.

Then, navigate to the root of the project in a terminal and use the following command:

```console
poetry install --with dev,test,docs
```

This command will create a dedicated virtual enviroment for the project and install all the dependencies for development, testing and building the documentation.

## Managing dependencies

If you are only experimenting, use Pip to install the package. Let say you want to experiment with the `pymeshfix` library. Before adding it to the list of dependencies, you should simply install it using Pip. When you do that, be sure that you are in the virtual enviroment created by poetry.

If what you see in your terminal is like this,

```console
D:\Projects\SigmaEpsilon\sigmaepsilon.mesh>
```

you are not operating in the virtual enviroment and issing the `pip install pymeshfix` command would install the library globally. You don't want that.

What you can do is to use the following command:

```console
poetry shell
```

After this, you should see something similar to this:

```console
(sigmaepsilon-mesh-py3.10) D:\Projects\SigmaEpsilon\sigmaepsilon.mesh>
```

Now you are in the realms of the virtual enviroment created by Poetry for this project and you are ready to install the package you want.

```console
pip install pymeshfix
```

If you are certain that a dependency is required for the project, you can add it like this:

```console
poetry add pymeshfix --group test
```

This would add `pymeshfix` as an optional dependency and it would only be installed if an end user installed `sigmaepsilon.mesh` using the command

```console
pip install sigmaepsilon.mesh "[test]"
```

### Installing third-party libraries in development mode

Sometimes -usually when working with other pacakges from the `sigmaepsilon` ecosystem, you would want to install them in editable mode, so that your modifications doesn't get lost. The proper way to do this is by using the following command inside a Poetry shell session:

```console
pip install "-e ..\sigmaepsilon.math [test,dev]"
```

This assumes that `sigmaepsilon.math` is in the same folder as `sigmaepsilon.mesh`.

The command would install `sigmaepsilon.math` in editable mode with optional dependencies for testing and development. Usually this is more than what you need (you just want your occasional changes to sigmaepsilon.math to not get lost) and you are good with

```console
pip install -e ..\sigmaepsilon.math
```

### Compliance with third-party licenses

When you add a new dependency, you should also update the license file for third-party libraries using

```console
poetry run pip-licenses --format=plain --output-file=THIRD-PARTY-LICENSES --order=license
```

or without `poetry run` if you are in a Poetry shell already.

This command creates the file `THIRD-PARTY-LICENSES`, where the libraries are ordered according to the type of license they are distributed under. Note that `pip-licenses` goes through all the currently installed packages in the enviroment and therefore it is crucial that the package should be installed without optional dependencies. You might need to delete the virtual enviroment

```console
poetry env remove the-name-of-the-enviroment
```

and install the library again using

```console
poetry install
```

If you don't know what is the name of your enviroment, you can use

```console
poetry env list
```

and it will list your enviroments, with the active one being highlighted.

After you updated the license file of third-party libraries, you should check if `sigmaepsilon.mesh` complies with the licenses of direct and indirect dependencies.

### Tracing back dependencies

If you wonder how a specific library ended up in the license file, you can use `pipdeptree`.

```console
poetry shell
pip install pipdeptree
```

Then to figure out how for instance the `triangle` library ended up in the licenses file, use this command:

```console
pipdeptree --reverse --packages triangle
```

and your output would include something like this:

```console
triangle==20230923
└── sectionproperties==3.2.0 [requires: triangle>=20230923,<20230924]
    └── sigmaepsilon.mesh==2.3.3 [requires: sectionproperties>2.1.3]
```

It shows, that triangle is a direct dependency of `sectionproperties`, which is a direct dependency of `sigmaepsilon.mesh`.

## Testing and Coverage

You find two bash scripts in the root folder, `run_tests_with_coverage.sh` and `run_tests_with_coverage_nojit.sh`. Use these to run the test, or open these files and use the contents to run the testing commands manually.

These commands create a coverage report in html format. Open `htmlcov/index.html` to see the results.
