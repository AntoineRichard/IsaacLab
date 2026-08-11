Added
^^^^^

* Added a set of greetings for the loading screen, some of them animated, chosen at random
  each run. Set ``ISAACLAB_LOADING_ANIM`` to a greeting's name to show that one every run, or
  to ``none`` to show none.
* Added :mod:`~isaaclab.app.anims`, which reads the shipped greetings from a compressed
  container. Build more with ``tools/gif2anim.py``, from an image, a GIF, or a procedural
  sprite, and inspect them with ``tools/anim_view.py``.

Fixed
^^^^^

* Fixed :meth:`~isaaclab.app.LoadingScreen.summary` measuring a greeting's width with its
  colour escapes included, which made any coloured greeting look too wide to fit and dropped
  it at every terminal size.
* Fixed the loading screen's elapsed clock reading a start time of exactly zero as "not
  started", which held the timer and any animation at zero for the whole run.
