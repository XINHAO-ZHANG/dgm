"""Exception types for the Apptainer client.

When the ``docker`` package is installed, these names are the real
``docker.errors`` classes so existing ``except docker.errors.ImageNotFound``
handlers keep working. The package does not need a running daemon.
"""

from __future__ import annotations

try:
    from docker.errors import APIError, BuildError, ImageNotFound, NotFound
except ImportError:  # pragma: no cover - exercised only without the docker package

    class APIError(Exception):
        def __init__(self, message, response=None, explanation=None):
            super().__init__(message)
            self.explanation = explanation or message

    class NotFound(APIError):
        pass

    class ImageNotFound(NotFound):
        pass

    class BuildError(Exception):
        def __init__(self, reason, build_log=""):
            super().__init__(reason)
            self.msg = reason
            self.build_log = build_log


__all__ = ["APIError", "BuildError", "ImageNotFound", "NotFound"]
