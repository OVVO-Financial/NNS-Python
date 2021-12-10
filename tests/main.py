if __name__ == "__main__":
    import sys
    import os
    import pytest

    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "NNS"))
    retcode = pytest.main()
