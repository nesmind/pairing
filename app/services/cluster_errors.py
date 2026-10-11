"""Why a node could not be added, removed or reset (routers map to 400 / 409)."""


class NodeNotAllowedError(ValueError):
    """Node cannot be added (400)."""


class NodeNotFreshError(Exception):
    """Node holds its own data (409)."""


class NodeResetRefusedError(Exception):
    """Node refused reset (409): still a voter, the original node, or not a joiner."""
