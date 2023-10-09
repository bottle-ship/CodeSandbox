import typing as t
from dataclasses import dataclass


@dataclass
class Item(object):
    name: str
    desc: str
    level: int
    lockable: bool
    has_checkbox: bool


@dataclass
class ExpandableItem(object):
    name: str
    desc: str
    level: int
    lockable: bool
    has_checkbox: bool
    sub_items: t.Tuple[t.Union[Item, 'ExpandableItem'], ...]


@dataclass
class RequirementItem(object):
    name: str
    package_name: str
    package_version: t.Optional[str]
    level: int
    lockable: bool = False
    has_checkbox: bool = True


@dataclass
class ExpandableRequirementItem(object):
    name: str
    desc: str
    level: int
    sub_items: t.Tuple[t.Union[RequirementItem, 'ExpandableRequirementItem'], ...]
    lockable: bool = False
    has_checkbox: bool = True
