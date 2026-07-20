Added
^^^^^

* Added OpenCV lens-distortion support to the Newton renderer: a camera cfg carrying an OpenCV
  pinhole (``k1..k6``, ``p1``, ``p2``, ``s1..s4``) or fisheye (``k1..k4``) distortion model on
  ``spawn.distortion`` is now rendered through the distortion instead of as a centered, square-pixel
  pinhole. The renderer inverts the OpenCV forward model per output pixel to build the distorted
  camera-space ray field, honoring the calibrated ``fx/fy/cx/cy`` intrinsics (including non-square
  focal lengths and an off-center principal point). With ``apply_lens_distortion=False`` the
  distortion coefficients are muted while the intrinsics are still applied, matching the RTX/OVRTX
  behavior.
