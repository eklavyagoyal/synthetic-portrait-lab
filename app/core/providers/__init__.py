"""Image-generation provider implementations.

Every provider subclasses :class:`app.core.providers.base.ImageProvider` and is
registered in :mod:`app.core.providers.registry`. The mock provider produces
real (tiny) PNG bytes locally and is used by the test-suite so that no test
ever touches a paid API.
"""
