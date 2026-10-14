"""Image processing operations, one module per syllabus unit.

Importing this package registers every operation in registry.OPERATIONS
(each function carries an @register(...) decorator).
"""

from operations import fundamentals, enhancement, morphology, edges, segmentation  # noqa: F401
