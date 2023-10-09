import typing as t

import requests
from packaging.version import parse as parse_version


def get_available_package_versions(package_name: str, python_version: str) -> t.List[str]:
    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return list()

    releases = data["releases"]
    compatible_versions = list()

    for version, release_info_list in releases.items():
        for release_info in release_info_list:
            requires_python = release_info.get("python_version", "")
            if not requires_python or python_version.replace(".", "") in requires_python:
                compatible_versions.append(version)
                break

    return sorted(compatible_versions, key=parse_version, reverse=True)
