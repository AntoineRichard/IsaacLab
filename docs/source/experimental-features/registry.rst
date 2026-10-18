.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _isaaclab-registry:

Isaac Lab Registry
==================

The Isaac Lab registry is a curated index of community task packages: projects that register Isaac Lab tasks, usually
created with the :ref:`template generator <template-generator>`. The ``isaaclab registry`` command finds these
packages, checks that they support your Isaac Lab release, installs them with uv, and helps authors publish their own.

.. attention::

   The registry is a prototype; its commands and generated fields may change. There is no public index yet, so point
   the command at one with ``ISAACLAB_REGISTRY_URL`` or ``--index`` (see :ref:`isaaclab-registry-index`). The
   registry bot is not running either: ``publish`` writes a registration request, or hands it to a local copy of the
   registry, but cannot submit it. The `IsaacLab-Registry <registry-repo_>`_ repository that this page links to is
   not public yet.

.. warning::

   Registry packages run third-party code at install time and every time Isaac Lab loads tasks. The registry pins and
   checks packages, but it does not review their code. Read the source of a package before you install it.


How the registry works
----------------------

The registry is a separate Git repository, `IsaacLab-Registry <registry-repo_>`_, run by a bot. Package code stays in
its authors' Git repositories or on a package index.

* An author asks the bot to register a release. The bot builds the release from its source and reads the version,
  Isaac Lab requirement, tasks, and RL libraries itself.
* The bot opens a pull request, runs its checks, and merges it after a waiting period. Any comment blocks the merge.
* Each release is pinned to a Git commit (the tag is only a label) or to the sha256 of a wheel, so an install gets
  exactly the code the registry checked.
* The bot compiles the registry into a single ``index.json`` file, which also holds a blocklist. ``isaaclab registry``
  downloads that file on every command and refuses blocklisted packages.

The `registry README <registry-repo_>`_ describes the checks, the waiting periods, and the rest of the bot's policy.

.. _registry-repo: https://github.com/isaac-sim/IsaacLab-Registry


Find packages
-------------

.. code-block:: bash

   uv run isaaclab registry list                # packages compatible with your Isaac Lab
   uv run isaaclab registry list --all          # also incompatible and end-of-life packages
   uv run isaaclab registry search locomotion   # match names, descriptions, tags, tasks, and categories
   uv run isaaclab registry info dummy-walk     # a package, its releases, and their compatibility

``list`` and ``search`` show the release that ``install`` would pick for each package:

.. code-block:: text

   +---------------------------------------------------------------------------------------------------------------------------------+
   |                                         isaaclab-community: packages for Isaac Lab 3.0.0                                        |
   +-------------+---------+--------------+-------+------------+---------------+-----------+-----------------------------------------+
   | Name        | Version | Category     | Tasks | Status     | Compatibility | Installed | Description                             |
   +-------------+---------+--------------+-------+------------+---------------+-----------+-----------------------------------------+
   | dummy-reach | 0.1.0   | manipulation | 1     | maintained | ok            |           | Isaac Lab environments for dummy_reach. |
   | dummy-walk  | 0.2.0   | locomotion   | 1     | maintained | ok            |           | Isaac Lab environments for dummy_walk.  |
   +-------------+---------+--------------+-------+------------+---------------+-----------+-----------------------------------------+
   [INFO] 1 incompatible or end-of-life package(s) hidden; use --all to show them.

``info`` lists every release with its source, Isaac Lab range, tasks, and RL libraries, and marks the one that
``install`` would pick with ``*``.


Install and use a package
-------------------------

.. code-block:: bash

   uv run isaaclab registry install dummy-walk

The command picks the newest compatible release, prints what it is about to install and where, and asks for
confirmation:

.. code-block:: text

   dummy-walk 0.2.0: Isaac Lab environments for dummy_walk.
     source       git https://github.com/alice/dummy-walk @ e4663a1756e0 (v0.2.0)
     license      BSD-3-Clause
     isaaclab     ~=3.0.0 (installed release: 3.0.0)
     tasks        DummyWalk-Walk-Cartpole
     requirement  dummy-walk @ git+https://github.com/alice/dummy-walk@e4663a1756e07bac3c54130e2d3c84af6d700dba
   Target: project /home/dave/my_robot_lab (dependency group 'isaaclab-registry', uv add)
   [WARNING] Registry packages run third-party code at install time and every time Isaac Lab loads tasks. Review the source.
   Install dummy-walk? [y/N]

Pass ``NAME==VERSION`` to install a specific release, ``--yes`` to skip the confirmation, or ``--dry-run`` to see what
uv would do without changing anything. The installed tasks then work like any other task:

.. code-block:: bash

   uv run isaaclab list_envs --keyword DummyWalk
   uv run isaaclab train --rl_library rsl_rl --task DummyWalk-Walk-Cartpole

Inside a project that registers its own tasks, ``list_envs`` shows them together with the tasks of the installed task
packages; pass ``--all`` to include Isaac Lab's tasks as well. Run ``uv run isaaclab registry uninstall dummy-walk`` to
remove a package. The dependencies that only it needed stay installed.

``uv run isaaclab registry installed`` reports the installed task packages and how they relate to the index:

.. code-block:: text

   Target: project /home/dave/my_robot_lab (dependency group 'isaaclab-registry', uv add)
   +--------------+---------+----------+--------------+--------------------------------+
   | Name         | Version | Status   | Commit       | Note                           |
   +--------------+---------+----------+--------------+--------------------------------+
   | dummy-walk   | 0.2.0   | registry | e4663a1756e0 |                                |
   | my-robot-lab | 0.1.0   | local    | -            | file:///home/dave/my_robot_lab |
   +--------------+---------+----------+--------------+--------------------------------+

* ``registry``: the installed commit or wheel is a registered release.
* ``outdated (X available)``: a newer compatible release exists. Install the package again to get it.
* ``blocked``: the registry blocklists the package or the installed release. Uninstall it.
* ``drift``: the package is in the registry, but the installed commit or wheel is not one of its releases.
* ``local``: an editable or local-directory install, such as your own project. The registry never changes it.
* ``unmanaged``: the package is not in the registry, or the index could not be read.

.. _isaaclab-registry-modes:

Where packages are installed
----------------------------

``install`` and ``uninstall`` choose their target from the environment they run in. In the environment of your own uv
project, which ``uv run`` selects from the project directory, they edit that project. Anywhere else, they change the
running environment. The ``Target:`` line names the choice before anything changes.

.. list-table::
   :widths: 24 40 36
   :header-rows: 1

   * - You run Isaac Lab from
     - ``install`` changes
     - A later ``uv sync``
   * - Your own uv project, such as one created with ``isaaclab --new``
     - ``uv add --group isaaclab-registry`` records the package in ``pyproject.toml`` and ``uv.lock``, then installs
       it. Commit both files to share the package with collaborators.
     - Keeps the package while ``isaaclab-registry`` is in the project's ``default-groups``. Generated projects list
       it, and ``install`` warns when a project does not.
   * - An Isaac Lab source checkout
     - ``uv pip install`` adds the package to the checkout's environment. The checkout's ``pyproject.toml`` and
       ``uv.lock`` stay untouched.
     - Removes the package, which the checkout's lock file does not list. Run ``install`` again.
   * - A virtual environment with the Isaac Lab wheel, outside a project
     - ``uv pip install`` adds the package to that environment.
     - Does not apply.

A project whose name starts with ``isaaclab`` counts as part of Isaac Lab and gets the environment target; rename it
to get project installs (the registry refuses such names anyway).

The registry never replaces or removes the installed Isaac Lab, an editable or local install such as your own project,
or an installed package that registers no tasks. In an environment, ``install`` pins every installed Isaac Lab package
and keeps every other installed package at its version, so it changes only the packages it names and adds their new
dependencies; uv fails when a package needs another version of one of them. In a project, ``install`` refuses a
package that would change the Isaac Lab locked in ``uv.lock``. Do not install through ``uvx``: its environments are
temporary.

.. _isaaclab-registry-compatibility:

Compatibility with your Isaac Lab
---------------------------------

Each release declares the Isaac Lab releases it supports with a bounded range, such as ``~=3.0.0``, and a Python range.
The command compares them with your Isaac Lab release: the ``VERSION`` file of a source checkout, or the version of
the installed wheel. In a checkout, the ``isaaclab`` package has an unrelated version, such as 37.2.0.

* ``install NAME`` picks the newest release that is compatible, not yanked, and not blocklisted. ``list`` hides the
  packages that have none, and end-of-life packages.
* ``install NAME==VERSION`` can also install a yanked release, with a warning.
* A blocklisted package or release is always refused. An end-of-life package installs with a warning.
* An incompatible release is refused unless you pass ``--force``:

  .. code-block:: text

     [ERROR] dummy-future 0.1.0 is not compatible: it needs isaaclab ~=4.0 (have 3.0.0). Pass --force to install it anyway.

A project generated from a source checkout resolves every ``isaaclab`` requirement to that checkout through an
override in its ``pyproject.toml``. uv therefore no longer checks the Isaac Lab range of a registry package, and
``install`` is the only check. The override also drops the Isaac Lab extras that a package requests: select them with
``uv run --group <extra>``, or add them to the project's ``isaaclab-dev`` dependency group.

.. _isaaclab-registry-publish:

Publish your project
--------------------

A project can be registered when:

* its ``pyproject.toml`` declares a static name and version, and an ``[project.entry-points."isaaclab.tasks"]`` table;
* its ``[project].dependencies`` require ``isaaclab`` with an upper bound, such as ``isaaclab[rsl-rl]~=3.0.0``, and
  neither its dependencies nor its extras require an ``isaaclab-*`` package such as ``isaaclab-dev`` (dependency groups
  are not published, so the ``isaaclab-dev`` group of a project generated from a checkout is fine);
* its code is in a public Git repository, or its wheel is on a package index;
* it has a license, a name that is not reserved (such as ``isaaclab*``, ``isaacsim*``, or ``omni*``), and task IDs that
  neither Isaac Lab nor another package registers.

``publish`` refuses a project that fails the first two conditions, or whose tag is not pushed; the registry bot checks
them all again. Projects generated with ``isaaclab --new`` meet the first two conditions and ship a BSD-3-Clause
``LICENSE``. Describe the package in the ``[tool.isaaclab.registry]`` table of ``pyproject.toml``:

.. code-block:: toml

   [tool.isaaclab.registry]
   category = "locomotion"
   tags = ["cartpole", "rsl-rl"]
   maintainers = ["alice"]
   media = { image = "https://example.com/dummy-walk.png" }

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Field
     - Value
   * - ``category``
     - One of ``locomotion``, ``manipulation``, ``navigation``, ``humanoid``, ``dexterous``, ``multi-agent``, or
       ``other``.
   * - ``tags``
     - Up to 15 tags made of lowercase letters, digits, and hyphens.
   * - ``maintainers``
     - The GitHub handles of the people who may register new versions. Include your own.
   * - ``media``
     - Optional ``image`` and ``video`` URLs.
   * - ``description``, ``homepage``
     - Optional. They default to the ``[project]`` description and ``[project.urls] Homepage``.
   * - ``status``, ``status_description``
     - Optional: ``developed``, ``maintained`` (the default), ``unmaintained``, or ``end-of-life``, and why.

Commit, tag the release as ``v<version>``, push the tag, and run ``publish`` from the project directory with your
GitHub handle (``--as`` defaults to the first maintainer):

.. code-block:: bash

   git tag v0.1.0
   git push origin HEAD v0.1.0
   uv run isaaclab registry publish --as alice

``publish`` looks up the tag (``--ref`` names another one) on the ``origin`` remote, or on the repository given with
``--repository``, and prints the registration request that the registry bot reads (``--output FILE`` writes it to a
file):

.. code-block:: toml

   schema = 1
   name = "dummy-walk"
   requested_by = "alice"

   [source]
   type = "git"
   repository = "https://github.com/alice/dummy-walk"
   subdirectory = ""
   ref = "v0.1.0"
   commit = "e139f6bbf503f6944d013f15aae5a1c861f8dffb"

   [package]
   # The [tool.isaaclab.registry] fields.

What happens next
~~~~~~~~~~~~~~~~~

There is no way to submit the request to a public registry yet. If you have access to
`IsaacLab-Registry <registry-repo_>`_, which is not public yet, clone it and pass the clone with ``--registry``:
``publish`` then hands the request to the bot of that clone, which opens a simulated pull request and prints the result
of each check. With ``--smoke``, the bot also imports the package with your Isaac Lab and checks that its tasks
register:

.. code-block:: bash

   uv run isaaclab registry publish --as alice --registry path/to/IsaacLab-Registry --smoke

To register a new version, raise ``version`` in ``pyproject.toml``, then commit, tag, push, and run ``publish`` again.
To register a wheel that is already on a package index instead of a Git commit, pass ``--pypi``, plus ``--index URL``
for a package index other than PyPI; the registry then pins the wheel's sha256. The `registry README <registry-repo_>`_
explains how to yank a release or widen its Isaac Lab range.

.. _isaaclab-registry-index:

Use another index
-----------------

The command reads the index from ``--index`` (placed before the subcommand), else from the ``ISAACLAB_REGISTRY_URL``
environment variable, else from the default location, a placeholder in this prototype. Each takes an ``index.json``
URL or file, or a registry clone, whose ``dist/index.json`` (written by the bot's ``build`` command) it reads:

.. code-block:: bash

   export ISAACLAB_REGISTRY_URL=https://example.com/registry/index.json
   uv run isaaclab registry --index path/to/index.json list

.. _isaaclab-registry-troubleshooting:

Troubleshooting
---------------

**A package is not compatible**
   Run ``uv run isaaclab registry info <name>`` to see the Isaac Lab range of each release. Use a matching Isaac Lab
   release, or pass ``--force`` to install the package anyway, at your own risk.

**The command refuses a package, uv cannot resolve it, or it is gone after uv sync**
   See :ref:`isaaclab-registry-modes`. In a source checkout, install the package again after ``uv sync``, and prefer
   ``uv run --extra <extra>``, which keeps it, over ``uv sync --extra <extra>``.

**"uses skrl, which is not installed here"**
   The package's agents use an RL library that the environment lacks. Run its tasks with the matching Isaac Lab
   extra, for example ``uv run --extra skrl isaaclab train ...``, or with ``uv run --group skrl ...`` in a project
   generated from a source checkout.

**Hash mismatch, or "not the wheel that was registered"**
   The wheel at the registered URL is no longer the file that the registry checked, so the install stops before it
   changes anything; in a project, ``pyproject.toml`` and ``uv.lock`` return to their previous content. Report it to
   the registry maintainers.

**A task is missing, or a task entry point failed to load**
   Isaac Lab skips a task package that fails to import and logs a warning that names it. Check its status with
   ``uv run isaaclab registry installed``, then install a compatible release or uninstall it.
