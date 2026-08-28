Added
^^^^^

* Added a ``rev`` argument to :func:`~isaaclab.utils.assets.retrieve_git_asset_path`, pinning an
  asset repository to a revision instead of tracking its default branch. A pinned remote repository
  is fetched by revision and cached separately from the same repository at another revision, so
  assets that are regenerated upstream stay reproducible. A pinned local checkout is only verified,
  never moved, so retrieving an asset cannot rewrite someone's working tree.
