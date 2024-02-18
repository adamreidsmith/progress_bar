import time
import sys
import os
from typing import Optional, Any
from collections.abc import Iterable, Generator, Callable


def handle_pbar_error(method):
    '''A decorator to add error handling to NestedPBar class methods'''

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
        options_dict: dict,
    ) -> None:

        # Validate the iterable
        try:
            self._iterator = iter(iterable)
        except TypeError:
            raise TypeError(f'\'{type(iterable).__name__}\' object is not iterable')

        # Compute the length if possible
        self._len = None
        try:
            self._len = len(iterable)
        except TypeError:
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
    _instance: 'NPB' = None  # Stores the running class instance
    _iterators: list[PBarIter] = []  # Stores the itrables whose progress is being evaluated
    _pbar_lines_written: int = 0  # Stores the number of progress bar lines currently written

    def __new__(cls, *args, **kwargs) -> 'NPB':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @handle_pbar_error
    def __init__(
        self,
        iterable,
        length: Optional[int] = None,
        desc: Optional[str] = None,
        fill_char: str = '█',
        counter: bool = True,
        timer: bool = True,
        rate: bool = True,
        avg_rate: bool = False,
        update_interval: float = 0.05,
    ) -> None:
        iterator = PBarIter(
            iterable=iterable,
            length=length,
            desc=desc,
            fill_char=fill_char,
            options_dict={'counter': counter, 'timer': timer, 'rate': rate, 'avg_rate': avg_rate},
        )
        self._iterators.append(iterator)

        # Validate update interval
        try:
            self._update_interval = float(update_interval)
        except Exception:
            raise TypeError('\'update_interval\' must be a float')
        self._last_update = float('-inf')

    def __iter__(self) -> 'NPB':
        return self

    @handle_pbar_error
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
            time_delta = self._iterators[-1]._it_time_delta
            curr_time = time.perf_counter()
            if curr_time - self._last_update >= self._update_interval or raise_se:
                self._last_update = curr_time
                self._write_pbars()
            if raise_se:
                cls._iterators.pop()
                if not cls._iterators:
                    self._reset()
                raise StopIteration

    def __del__(self) -> None:
        self._reset()

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

        pbar_space = os.get_terminal_size().columns - len(prefix) - bool(prefix) - len(suffix) - bool(suffix)
        pbar = self._get_pbar(iterator, pbar_space)
        return ' '.join(filter(bool, (prefix, pbar, suffix)))

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

        cls._instance = None
        cls._iterators = []
        cls._pbar_lines_written = 0

    def cancel(self) -> None:
        '''Cancel the current progress bar instance'''

        cls = self.__class__
        if len(cls._iterators) <= 1:
            self._reset()
        else:
            cls._iterators.pop()


if __name__ == '__main__':
    from tqdm import tqdm

    t = time.perf_counter()
    for i in NPB(range(30), desc='Master bar'):
        for j in NPB(range(15), desc='Sub Bar'):
            for k in NPB(range(10), desc='Sub Sub Bar'):
                time.sleep(0.0005)
    a = time.perf_counter() - t

    t = time.perf_counter()
    for i in tqdm(range(30), desc='Master bar'):
        for j in tqdm(range(15), desc='Sub Bar'):
            for k in tqdm(range(10), desc='Sub Sub Bar'):
                time.sleep(0.0005)
    b = time.perf_counter() - t

    t = time.perf_counter()
    for i in range(30):
        for j in range(15):
            for k in range(10):
                time.sleep(0.0005)
    c = time.perf_counter() - t

    print(f'NPB: {a}')
    print(f'tqdm: {b}')
    print(f'None: {c}')
