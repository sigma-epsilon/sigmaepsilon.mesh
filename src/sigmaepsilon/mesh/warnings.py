from sigmaepsilon.core.warning import SigmaEpsilonWarning


class SigmaEpsilonMeshImportWarning(SigmaEpsilonWarning):
    """
    Warning that occurs during file imports.
    """
    def __init__(self, message: str):
        pre = "SigmaEpsilon Import Warning: "
        self.message = pre + message