"""
On Editor/Engine start, the ue_site module is tried for import. You should place initialization code there.
If the module cannot be imported, you will get a (harmful) message in the logs.

This file is also necessary for packaged builds
"""

import ensure_requirements


def main():
    ensure_requirements.ensure_requirements()


main()
