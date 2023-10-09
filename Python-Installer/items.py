from base import Item, ExpandableItem, RequirementItem, ExpandableRequirementItem
from utils import get_available_package_versions


AVAILABLE_PYTHON_VERSIONS = ("3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "3.11")


PYTHON_VERSION = ExpandableItem(
    name="PYTHON_VERSION",
    desc="Python version",
    level=0,
    lockable=False,
    has_checkbox=True,
    sub_items=tuple(
        Item(
            name=f"PYTHON_VERSION-{py_version}",
            desc=py_version,
            level=1,
            lockable=False,
            has_checkbox=True
        )
        for py_version in AVAILABLE_PYTHON_VERSIONS
    )
)


PYTHON_INSTALL_PATH = Item(
    name="PYTHON_INSTALL_PATH",
    desc="Enter python install path",
    level=0,
    lockable=False,
    has_checkbox=False
)


PACKAGE_REQUIREMENTS = ExpandableItem(
    name="PACKAGE_REQUIREMENTS",
    desc="Python package requirements",
    level=0,
    lockable=True,
    has_checkbox=False,
    sub_items=()
)


PACKAGE_REQUIREMENTS_SUB_ITEMS = {}
for py_version in AVAILABLE_PYTHON_VERSIONS:
    PACKAGE_REQUIREMENTS_SUB_ITEMS[py_version] = list()

    tf_versions = get_available_package_versions(package_name="tensorflow", python_version=py_version)
    PACKAGE_REQUIREMENTS_SUB_ITEMS[py_version].append(
        ExpandableRequirementItem(
            name="TENSORFLOW",
            desc="tensorflow",
            level=1,
            sub_items=tuple(
                [
                    RequirementItem(
                        name=f"TENSORFLOW#{tf_version}",
                        package_name="tensorflow",
                        package_version=tf_version,
                        level=2
                    )
                    for tf_version in tf_versions
                ]
            )
        )
    )

    torch_versions = get_available_package_versions(package_name="torch", python_version=py_version)
    PACKAGE_REQUIREMENTS_SUB_ITEMS[py_version].append(
        ExpandableRequirementItem(
            name="TORCH",
            desc="torch",
            level=1,
            sub_items=tuple(
                [
                    RequirementItem(
                        name=f"TORCH#{torch_version}",
                        package_name="torch",
                        package_version=torch_version,
                        level=2
                    )
                    for torch_version in torch_versions
                ]
            )
        )
    )

    PACKAGE_REQUIREMENTS_SUB_ITEMS[py_version].append(
        RequirementItem(name="PANDAS", package_name="pandas", package_version=None, level=1)
    )
    PACKAGE_REQUIREMENTS_SUB_ITEMS[py_version].append(
        RequirementItem(name="SKLEARN", package_name="scikit-learn", package_version=None, level=1)
    )


INSTALL = Item(
    name="INSTALL",
    desc="Install",
    level=0,
    lockable=True,
    has_checkbox=False
)
