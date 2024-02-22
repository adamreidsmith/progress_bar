import time
import sys
import os
from typing import Optional, Any
from collections.abc import Iterable, Generator, Callable


_VALID_COLORS = {
    'black': ('\033[30m', '\033[40m'),
    'red': ('\033[31m', '\033[41m'),
    'green': ('\033[32m', '\033[42m'),
    'yellow': ('\033[33m', '\033[43m'),
    'blue': ('\033[34m', '\033[44m'),
    'magenta': ('\033[35m', '\033[45m'),
    'cyan': ('\033[36m', '\033[46m'),
    'white': ('\033[37m', '\033[47m'),
}
_RESET_TF = '\033[0m'
_RAINBOW = tuple(_VALID_COLORS[c][0] for c in ('red', 'yellow', 'green', 'cyan', 'blue', 'magenta'))

_NPB_DEFAULT_CLS_ATT = {
    '_instance': None,
    '_iterators': (),  # Using a tuple for immutability
    '_pbar_lines_written': 0,
    '_update_interval': None,
    '_default_update_interval': 0.05,
}


def handle_NPB_error(method):
    '''A decorator to add error handling to NPB class methods'''

    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except StopIteration:
            raise
        except Exception:
            self._reset()
            raise

    return wrapper


class PBarIter:
    def __init__(
        self,
        iterable: Iterable,
        length: Optional[int],
        desc: Optional[str],
        fill_char: str,
        ncols: Optional[int],
        text_color: Optional[str],
        bg_color: Optional[str],
        rainbow: bool,
        options_dict: dict,
    ) -> None:

        # Validate the iterable
        try:
            self._iterator = iter(iterable)
        except (TypeError, AttributeError):
            raise TypeError(f'\'{type(iterable).__name__}\' object is not iterable')

        # Compute the length if possible
        self._len = None
        try:
            self._len = len(iterable)
        except (TypeError, AttributeError):
            if length is not None:
                try:
                    self._len = int(length)
                except Exception:
                    raise TypeError('\'length\' must be a non-negative integer')

        # Validate the description
        self._desc = desc
        if self._desc is not None:
            if not isinstance(self._desc, str):
                raise TypeError('\'desc\' must be a string')
            if len(self._desc) == 0:
                self._desc = None

        # Validate the fill char
        self._fill_char = fill_char
        if not isinstance(self._fill_char, str):
            raise TypeError('\'fill_char\' must be a string of length one')
        if len(self._fill_char) != 1:
            raise ValueError('\'fill_char\' must be a string of length one')

        # Validate ncols
        if ncols is not None:
            try:
                ncols = int(ncols)
            except Exception:
                raise TypeError('If specified, \'ncols\' must be an integer')
        self._ncols = ncols

        # Validate text_color
        if text_color is not None and text_color not in _VALID_COLORS:
            raise ValueError(f'\'text_color\' must be a string in {set(_VALID_COLORS.keys())}')
        self._text_color = _VALID_COLORS[text_color][0] if text_color is not None else None

        # Validate bg_color
        if bg_color is not None and bg_color not in _VALID_COLORS:
            raise ValueError(f'\'bg_color\' must be a string in {set(_VALID_COLORS.keys())}')
        self._bg_color = _VALID_COLORS[bg_color][1] if bg_color is not None else None

        # Validate rainbow
        self._rainbow = rainbow
        if not isinstance(self._rainbow, bool):
            raise TypeError('\'rainbow\' must be a boolean')

        # Validate the options dictionary
        self._options_dict = options_dict
        if any(not isinstance(opt, bool) for opt in options_dict.values()):
            raise TypeError('Only boolean values are accepted for option arguments')

        self._time_of_current_it: float = None
        self._it_time_delta: float = None
        self._start_time: float = None
        self._current_index: int = -1

    def __len__(self) -> int | None:
        return self._len

    def __iter__(self) -> 'PBarIter':
        self._current_index = -1
        self._it_time_delta = None
        self._start_time = None
        self._time_of_current_it = None
        return self

    def __next__(self) -> Any:
        raise_se = False
        try:
            next_item = next(self._iterator)
        except StopIteration:
            raise_se = True

        curr_time = time.perf_counter()
        if self._time_of_current_it is not None:
            self._it_time_delta = curr_time - self._time_of_current_it
        self._time_of_current_it = curr_time
        if self._start_time is None:
            self._start_time = curr_time

        self._current_index += 1

        if raise_se:
            raise StopIteration

        return next_item


class NPB:
    # Stores the running class instance
    _instance: Optional['NPB'] = _NPB_DEFAULT_CLS_ATT['_instance']
    # Stores the itrables whose progress is being evaluated
    _iterators: tuple[PBarIter, ...] = _NPB_DEFAULT_CLS_ATT['_iterators']
    # Stores the number of progress bar lines currently written
    _pbar_lines_written: int = _NPB_DEFAULT_CLS_ATT['_pbar_lines_written']
    # How often we update the progress bar and its default value
    _update_interval: Optional[float] = _NPB_DEFAULT_CLS_ATT['_update_interval']
    _default_update_interval: float = _NPB_DEFAULT_CLS_ATT['_default_update_interval']

    def __new__(cls, iterable, /, *args, **kwargs) -> 'NPB':
        # If disable is True, return the iterable
        disable = kwargs.get('disable', False)
        if isinstance(disable, bool) and disable:
            return iterable
        # If a running class instance already exists,
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @handle_NPB_error
    def __init__(
        self,
        iterable,  # Passed to PBarIter
        /,
        *,
        length: Optional[int] = None,  # Passed to PBarIter
        desc: Optional[str] = None,  # Passed to PBarIter
        fill_char: str = '█',  # Passed to PBarIter
        update_interval: float = _default_update_interval,
        disable: bool = False,
        ncols: Optional[int] = None,  # Passed to PBarIter
        text_color: Optional[str] = None,  # Passed to PBarIter
        bg_color: Optional[str] = None,  # Passed to PBarIter
        rainbow: bool = False,  # Passed to PBarIter
        # Options (all passed to PBarIter):
        counter: bool = True,
        timer: bool = True,
        rate: bool = True,
        avg_rate: bool = False,
    ) -> None:
        cls = self.__class__

        iterator = PBarIter(
            iterable=iterable,
            length=length,
            desc=desc,
            fill_char=fill_char,
            ncols=ncols,
            text_color=text_color,
            bg_color=bg_color,
            rainbow=rainbow,
            options_dict={'counter': counter, 'timer': timer, 'rate': rate, 'avg_rate': avg_rate},
        )

        cls._iterators = (*self._iterators, iterator)

        # Validate update_interval
        try:
            ui = float(update_interval)
            if ui != cls._default_update_interval:
                cls._update_interval = ui
        except Exception:
            raise TypeError('\'update_interval\' must be a float')
        self._last_update = float('-inf')

        # Validate disable
        # This parameter is handled in the __new__ method and does nothing from here on.
        # However, we still type check it.
        if not isinstance(disable, bool):
            raise TypeError('\'disable\' must be a boolean')

    def __iter__(self) -> 'NPB':
        return self

    @handle_NPB_error
    def __next__(self) -> Any:
        cls = self.__class__
        raise_se = False
        try:
            next_item = next(cls._iterators[-1])
            return next_item
        except StopIteration:
            raise_se = True
        finally:
            # Always executed
            curr_time = time.perf_counter()
            ui = cls._update_interval if cls._update_interval is not None else cls._default_update_interval
            if curr_time - self._last_update >= ui or raise_se:
                self._last_update = curr_time
                self._write_pbars()
            if raise_se:
                cls._iterators = cls._iterators[:-1]
                if not cls._iterators:
                    self._reset()
                raise StopIteration

    def _write_pbars(self) -> None:
        cls = self.__class__
        n_iterators = len(cls._iterators)

        extra_lines_needed = n_iterators - cls._pbar_lines_written
        if extra_lines_needed > 0:
            sys.stdout.write('\n' * extra_lines_needed)
            cls._pbar_lines_written += extra_lines_needed

        sys.stdout.write(f'\r\033[{cls._pbar_lines_written}A')  # Move the cursor to the start of the top pbar
        for i, iterator in enumerate(cls._iterators):
            # Clear current line after cursor, write the pbar, and move cursor down one and to the start of the line
            sys.stdout.write(f'\033[K{self._get_pbar_str(iterator)}\033[1B\r')
            if i == len(cls._iterators) - 1:
                # Last iterator has been written, so clear lines below
                for _ in range(cls._pbar_lines_written - i - 1):
                    sys.stdout.write('\033[K\033[1B')  # Clear current line after cursor and move cursor down 1
        sys.stdout.flush()

    def _get_options(self, iterator: PBarIter) -> Generator[Callable[[PBarIter], str], None, None]:
        opt_dict = iterator._options_dict
        if opt_dict.get('counter'):
            yield self._get_count
        if opt_dict.get('timer'):
            yield self._get_elapsed_remaining
        if opt_dict.get('rate'):
            yield self._get_iteration_rate
        if opt_dict.get('avg_rate'):
            yield self._get_avg_rate

    def _get_pbar_str(self, iterator: PBarIter) -> str:
        prefix = self._get_desc(iterator)
        suffix = ' '.join(filter(bool, (f(iterator) for f in self._get_options(iterator))))

        width = os.get_terminal_size().columns if iterator._ncols is None else iterator._ncols
        pbar_space = width - len(prefix) - bool(prefix) - len(suffix) - bool(suffix)
        pbar = self._get_pbar(iterator, pbar_space)
        pbar_str = ' '.join(filter(bool, (prefix, pbar, suffix)))

        # Set the color of the progress bar
        if iterator._rainbow:
            pbar_str = ''.join(f'{_RAINBOW[i % len(_RAINBOW)]}{c}' for i, c in enumerate(pbar_str)) + _RESET_TF
        elif iterator._text_color is not None:
            pbar_str = f'{iterator._text_color}{pbar_str}{_RESET_TF}'
        if iterator._bg_color is not None:
            pbar_str = f'{iterator._bg_color}{pbar_str}{_RESET_TF}'

        return pbar_str

    @staticmethod
    def _get_desc(iterator: PBarIter) -> str:
        if iterator._desc is None:
            return ''
        return f'{iterator._desc}:'

    @staticmethod
    def _get_count(iterator: PBarIter) -> str:
        if iterator._len is None:
            return f'{iterator._current_index}it'
        return f'{iterator._current_index}/{iterator._len}'

    def _get_elapsed_remaining(self, iterator: PBarIter) -> str:
        elapsed_time = proj_time = ''
        if iterator._it_time_delta is not None:
            elapsed_time = self._format_time(iterator._time_of_current_it - iterator._start_time)
            if iterator._len is not None:
                proj_time = self._format_time((iterator._len - iterator._current_index) * iterator._it_time_delta)

        elapsed_time = elapsed_time or self._format_time(0)
        proj_time = proj_time or '?'

        return f'{elapsed_time}<{proj_time}'

    @staticmethod
    def _get_iteration_rate(iterator: PBarIter) -> str:
        if iterator._it_time_delta is None:
            rate = '?'
        else:
            if iterator._it_time_delta < 1.0:
                rate = f'{1.0 / iterator._it_time_delta:.2f}it/s'.rjust(9)
            else:
                rate = f'{iterator._it_time_delta:.2f}s/it'.rjust(9)
        return rate

    @staticmethod
    def _get_avg_rate(iterator: PBarIter) -> str:
        if iterator._it_time_delta is None:
            rate = '?'
        else:
            rate = (iterator._time_of_current_it - iterator._start_time) / iterator._current_index
            if rate < 1.0:
                rate = f'{1.0 / rate:.2f}it/s'.rjust(9)
            else:
                rate = f'{rate:.2f}s/it'.rjust(9)
        return rate

    @staticmethod
    def _get_pbar(iterator: PBarIter, space_avail: int) -> str:
        if iterator._len is None:
            return ''

        prop_done = iterator._current_index / iterator._len
        percent = f'{prop_done:.0%}'

        pbar_space = space_avail - len(percent) - 2  # -2 for | chars

        if pbar_space > 0:
            pbar_str = (iterator._fill_char * int(prop_done * pbar_space)).ljust(pbar_space)
            return f'{percent}|{pbar_str}|'
        elif pbar_space > -3:
            return percent
        else:
            return ''

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes, secs = divmod(round(seconds), 60)
        hours, mins = divmod(minutes, 60)
        if hours > 0:
            return f'{hours}:{mins:02d}:{secs:02d}'
        return f'{minutes:02d}:{secs:02d}'

    @classmethod
    def _reset(cls) -> None:
        '''Reset the class state so that a new instance can be created'''

        for att in _NPB_DEFAULT_CLS_ATT.keys():
            setattr(cls, att, _NPB_DEFAULT_CLS_ATT[att])

    def __del__(self) -> None:
        self._reset()

    def cancel(self) -> None:
        '''Cancel the current progress bar instance'''

        cls = self.__class__
        if len(cls._iterators) <= 1:
            self._reset()
        else:
            cls._iterators = cls._iterators[:-1]


def nrange(*args, **kwargs):
    '''Shortcut for NPB(range(*args), **kwargs)'''
    return NPB(range(*args), **kwargs)


if __name__ == '__main__':
    # from tqdm import tqdm

    # t = time.perf_counter()
    # for i in NPB(range(30), desc='Master bar'):
    #     for j in NPB(range(15), desc=f'Sub Bar {i}'):
    #         for k in NPB(range(10), desc=f'Sub Sub Bar {j}'):
    #             time.sleep(0.0005)
    # a = time.perf_counter() - t

    # t = time.perf_counter()
    # for i in NPB(range(30), desc='Master bar', rainbow=True):
    #     for j in NPB(range(15), desc=f'Sub Bar {i}', rainbow=True):
    #         for k in NPB(range(10), desc=f'Sub Sub Bar {j}', rainbow=True):
    #             time.sleep(0.0005)
    # a2 = time.perf_counter() - t

    # t = time.perf_counter()
    # for i in tqdm(range(30), desc='Master bar'):
    #     for j in tqdm(range(15), desc=f'Sub Bar {i}'):
    #         for k in tqdm(range(10), desc=f'Sub Sub Bar {j}'):
    #             time.sleep(0.0005)
    # b = time.perf_counter() - t

    # t = time.perf_counter()
    # for i in range(30):
    #     for j in range(15):
    #         for k in range(10):
    #             time.sleep(0.0005)
    # c = time.perf_counter() - t

    # print(f'NPB: {a}')
    # print(f'NPB rainbow: {a2}')
    # print(f'tqdm: {b}')
    # print(f'None: {c}')

    print(nrange(4))
