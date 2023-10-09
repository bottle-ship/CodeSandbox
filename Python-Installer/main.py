import curses
import curses.ascii
import sys
import typing as t
from collections import defaultdict

from base import (
    Item,
    ExpandableItem,
    RequirementItem,
    ExpandableRequirementItem
)
from items import (
    PYTHON_VERSION,
    PYTHON_INSTALL_PATH,
    PACKAGE_REQUIREMENTS,
    PACKAGE_REQUIREMENTS_SUB_ITEMS,
    INSTALL
)


class Screen(object):
    UP: int = -1
    DOWN: int = 1
    SPACE: int = 2

    def __init__(self, items: t.Tuple[t.Union[Item, ExpandableItem], ...]):
        self.items = items

        self._window = None
        self._max_lines = 0
        self._x_lim = 0
        self._y_lim = 0

        self._activated_items = list()
        self._req_package_items = list()

        self._top = 0
        self._bottom = len(self._activated_items)
        self._current_line = 0

        self._is_locked = defaultdict(lambda: False)
        self._is_expand = defaultdict(lambda: False)
        self._is_checked = defaultdict(lambda: False)

        self._python_version = None
        self._target_path = None
        self._pkg_requirements = dict()

        self._initialize()

    def run(self):
        try:
            self._input_stream()
        except KeyboardInterrupt:
            sys.exit()
        finally:
            self._window.keypad(False)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
            sys.exit()

    def _initialize(self):
        self._window = curses.initscr()
        curses.noecho()
        curses.cbreak()

        self._window.keypad(True)

        curses.curs_set(0)
        self._window.clear()

        y_max, x_max = self._window.getmaxyx()
        self._x_lim = x_max - 2
        self._y_lim = y_max - 1

        self._max_lines = curses.LINES - 5

        self._is_locked.update({item.name: item.lockable for item in self.items})

    def _draw_border_line(self):
        if self._x_lim > 0 and self._y_lim > 0:
            self._window.hline(0, 0, curses.ACS_HLINE, self._x_lim)
            self._window.hline(self._y_lim, 0, curses.ACS_HLINE, self._x_lim)

            self._window.vline(0, 0, curses.ACS_VLINE, self._y_lim)
            self._window.vline(0, self._x_lim, curses.ACS_VLINE, self._y_lim)

            self._window.addch(0, 0, curses.ACS_ULCORNER)
            self._window.addch(0, self._x_lim, curses.ACS_URCORNER)
            self._window.addch(self._y_lim, 0, curses.ACS_LLCORNER)
            self._window.addch(self._y_lim, self._x_lim, curses.ACS_LRCORNER)

    def _register_items(
            self,
            items: t.Tuple[
                t.Union[Item, ExpandableItem, RequirementItem, ExpandableRequirementItem],
                ...
            ]
    ):
        for item in items:
            self._activated_items.append(item)
            if isinstance(item, (ExpandableItem, ExpandableRequirementItem)) and self._is_expand[item.name]:
                self._register_items(item.sub_items)

    def _draw_menus(self):
        self._activated_items.clear()

        self._register_items(self.items)

        self._bottom = len(self._activated_items)

        for i, item in enumerate(self._activated_items[self._top:self._top + self._max_lines]):
            name = item.name
            if isinstance(item, RequirementItem):
                package_name = item.package_name
                package_version = item.package_version
                desc = package_name if package_version is None else package_version
            else:
                desc = item.desc
            level = item.level
            lockable = item.lockable
            has_checkbox = item.has_checkbox

            if lockable and self._is_locked[name]:
                desc = f"{desc} (Locked)"

            if has_checkbox:
                desc = f"[X] {desc}" if self._is_checked[name] else f"[ ] {desc}"

            if isinstance(item, (ExpandableItem, ExpandableRequirementItem)):
                space = self.SPACE + (self.SPACE * level)
                desc = f"- {desc}" if self._is_expand[name] else f"+ {desc}"
            else:
                space = self.SPACE * 2 + (self.SPACE * level)

            if name == "PYTHON_VERSION":
                if self._python_version is not None:
                    desc = f"{desc} - {self._python_version}"
            elif name == "PYTHON_INSTALL_PATH":
                if self._target_path is not None:
                    desc = f"Install python to \"{self._target_path}\""

            if i == self._current_line:
                self._window.addstr(i + 2, space, desc, curses.A_BOLD | curses.A_REVERSE)
            else:
                self._window.addstr(i + 2, space, desc)

        if isinstance(self._activated_items[self._current_line], (ExpandableItem, ExpandableRequirementItem)):
            self._window.addstr(self._y_lim - 1, self.SPACE, "Up/Down: Move | Left/Right: Expand | 'ESC': Exit")
        else:
            self._window.addstr(self._y_lim - 1, self.SPACE, "Up/Down: Move |  'Enter': Select | 'ESC': Exit")

    def _scroll(self, direction: int):
        next_line = self._current_line + direction

        if (direction == self.UP) and (self._top > 0 and self._current_line == 0):
            self._top += direction
            return

        if (direction == self.DOWN) and (next_line == self._max_lines) and (self._top + self._max_lines < self._bottom):
            self._top += direction
            return

        if (direction == self.UP) and (self._top > 0 or self._current_line > 0):
            if self._is_locked[self._activated_items[self._current_line].name]:
                self._scroll(direction + self.UP)
            else:
                self._current_line = next_line

            return

        if (direction == self.DOWN) and (next_line < self._max_lines) and (self._top + next_line < self._bottom):
            if self._is_locked[self._activated_items[next_line].name]:
                self._scroll(direction + self.DOWN)
            else:
                self._current_line = next_line

            return

    def _expand(self, expand: bool):
        current_item = self._activated_items[self._current_line]
        if isinstance(current_item, (ExpandableItem, ExpandableRequirementItem)):
            self._is_expand[current_item.name] = expand

    def _input_stream(self):
        while True:
            self._window.erase()
            self._draw_border_line()
            self._draw_menus()

            self._window.refresh()

            ch = self._window.getch()
            if ch == curses.KEY_UP:
                self._scroll(self.UP)
            elif ch == curses.KEY_DOWN:
                self._scroll(self.DOWN)
            elif ch == curses.KEY_LEFT:
                self._expand(False)
            elif ch == curses.KEY_RIGHT:
                self._expand(True)
            elif ch == ord("\n"):
                item = self._activated_items[self._current_line]
                if item.name.startswith("PYTHON_VERSION-"):
                    self._set_python_version()
                elif item.name == "PYTHON_INSTALL_PATH":
                    self._set_target_path()
                elif item.name == "INSTALL":
                    self._install()
                elif isinstance(item, RequirementItem):
                    self._set_pkg_requirements()

            elif ch == curses.ascii.ESC:
                sys.exit()

    def _update_req_packages(self):
        self._req_package_items.clear()

        self._req_package_items.extend(PACKAGE_REQUIREMENTS_SUB_ITEMS[self._python_version])

    def _set_is_locked(self):
        if self._python_version is not None and self._target_path is not None:
            self._is_locked[PACKAGE_REQUIREMENTS.name] = False
            self._is_locked[INSTALL.name] = False
        else:
            self._is_locked[PACKAGE_REQUIREMENTS.name] = True
            self._is_locked[INSTALL.name] = True

    def _set_python_version(self):
        item = self._activated_items[self._current_line]
        name = item.name
        python_version = item.desc
        if self._python_version is None:
            self._python_version = python_version
            self._is_checked[name] = True
            self._is_checked[PYTHON_VERSION.name] = True
            self._update_req_packages()
        else:
            if python_version == self._python_version:
                self._python_version = None
                self._is_checked[name] = False
                self._is_checked[PYTHON_VERSION.name] = False
                self._req_package_items.clear()
            else:
                self._is_checked[f"PYTHON_VERSION-{self._python_version}"] = False
                self._python_version = python_version
                self._is_checked[name] = True
                self._is_checked[PYTHON_VERSION.name] = True
                self._update_req_packages()

        for i, item in enumerate(self.items):
            if item.name == "PACKAGE_REQUIREMENTS":
                self.items[i].sub_items = tuple(self._req_package_items)
                break

        for package_name in self._pkg_requirements.keys():
            self._is_checked[package_name.upper()] = False
            package_version = self._pkg_requirements[package_name]
            if package_version is None:
                self._is_checked[f"{package_name.upper()}#{package_version}"] = False

        self._pkg_requirements.clear()

        self._set_is_locked()

    def _set_target_path(self):
        curses.echo()
        stdscr = curses.initscr()
        curses.curs_set(1)
        stdscr.clear()

        self._draw_border_line()

        stdscr.addstr(1, self.SPACE, "Enter Python Install Path")
        stdscr.addstr(self._y_lim - 1, self.SPACE, "'Enter': Finish")
        stdscr.refresh()
        self._target_path = stdscr.getstr(2, self.SPACE).decode("utf-8")
        curses.noecho()
        stdscr.clear()
        stdscr.refresh()

        self._set_is_locked()

    def _set_pkg_requirements(self):
        item = self._activated_items[self._current_line]
        name = item.name
        package_name = item.package_name
        package_version = item.package_version

        if self._is_checked[package_name.upper()]:
            if self._is_checked[name]:
                self._is_checked[name] = False
                self._is_checked[package_name.upper()] = False
                del self._pkg_requirements[package_name]
            else:
                for key in self._is_checked.keys():
                    if key.startswith(f"{package_name.upper()}#"):
                        self._is_checked[key] = False
                self._is_checked[name] = True
                self._is_checked[package_name.upper()] = True
                self._pkg_requirements[package_name] = package_version
        else:
            self._is_checked[name] = True
            self._is_checked[package_name.upper()] = True
            self._pkg_requirements[package_name] = package_version

    def ttt(self):
        print("!@#")

    def _install(self):
        curses.echo()
        stdscr = curses.initscr()
        stdscr.clear()
        stdscr.refresh()

        self._draw_border_line()

        stdscr.addstr(1, self.SPACE, "'Enter': Finish")

        self.ttt()
        print(sys.stdout.readlines())
        print("!!!")

        stdscr.addstr(self._y_lim - 1, self.SPACE, "'Enter': Finish")
        stdscr.refresh()

        stdscr.getstr(2, self.SPACE).decode("utf-8")

        curses.noecho()
        stdscr.clear()
        stdscr.refresh()


if __name__ == '__main__':
    Screen(items=(PYTHON_VERSION, PYTHON_INSTALL_PATH, PACKAGE_REQUIREMENTS, INSTALL)).run()
