import random
import curses
import bisect
import time
from enum import Enum

ModeNames = Enum('Modes', 'BASE_FEE PRIORITY_FEE')
mode_keys = [ord('1'), ord('2')]

class Block:
    # Relevant block data.
    def __init__(self, data):
        self.basefee
        self.fees = self.Fees()

    class Fees:
        def __init__(self):
            pass

def get_mode_from_key(keypress):
    mode = None
    if keypress == ord('1'):
        mode = ModeNames.BASE_FEE
    if keypress == ord('2'):
        mode = ModeNames.PRIORITY_FEE
    return mode

def define_axes(mode):
    price_str = 'nanoether per gas (Gwei per gas)'
    block_str = 'block number'
    if mode == ModeNames.BASE_FEE:
        return block_str, price_str, 'base fee'
    elif mode == ModeNames.PRIORITY_FEE:
        return block_str, price_str, 'priority fee'

class Mode:
    # The graph context (data, window display)

    def __init__(self, keypress):
        self.current = get_mode_from_key(keypress)
        ax = define_axes(self.current)
        self.x_name = ax[0]
        self.y_name = ax[1]
        self.graph_name = ax[2]
        self.data = None



class Keypress:
    # User input during program.
    def __init__(self, starting_key):
        self.key = None
        self.active = True
        self.current_mode = starting_key
        self.mode_changed = False
        
    def modify_mode(self):
        # Interprets mode selection when a number key is pressed.
        if self.key == self.current_mode:
            self.mode_changed = False
        else:
            self.current_mode = self.key
            self.mode_changed = True

    def read(self, win):
        # Reads last key pressed, detects changes.
        self.key = win.getch()
        if self.key == ord('q'):
            self.active = False

        if self.key in mode_keys:
            self.modify_mode()


class Interval:
    # Determines if it is the right time to get data.
    def __init__(self):
        self.sec_since_call = 0
        self.time = int(time.time())
        self.ready_to_call_block = False
    
    def reset(self):
        self.sec_since_call = 0
    
    def update(self):
        self.sec_since_call =  int(time.time()) - self.time
        if self.sec_since_call >= 2:
            self.ready_to_call_block = True
        else:
            self.ready_to_call_block = False

class Positions:
    # Coordinates of elements, in "(y, x)" where paired.
    def __init__(self, sc, win):
        self.border = 4
        self.get_fixed(sc, win)

    def get_fixed(self, sc, win):
        self.h, self.w = sc.getmaxyx()

        self.y_axis_tip = (self.border, self.border)
        self.y_axis_height = self.h - (2 * self.border) + 1
        self.x_axis_base = (self.h-self.border, self.border)
        self.x_axis_width = self.w - (2 * self.border)


def draw_axes(win, pos, mode):
    # Creates and labels graph axes.
    # x-axis
    win.hline(pos.x_axis_base[0], pos.x_axis_base[1], 
        '_', pos.x_axis_width)
    x_name = mode.x_name 
    win.addstr(pos.x_axis_base[0] + 2, 
        pos.x_axis_base[1] + pos.x_axis_width - len(x_name), mode.x_name)
    # y-axis
    win.vline(pos.y_axis_tip[0], pos.y_axis_tip[1], 
        '|', pos.y_axis_height)
    win.addstr(pos.y_axis_tip[0] - 2, pos.y_axis_tip[1] - 2, mode.y_name)
    # title
    win.addstr(pos.y_axis_tip[0] - 2, 
        pos.w - pos.border - len(mode.graph_name), mode.graph_name)
    return 


def draw_graph(sc, win, mode):
    # Gets positions of elements for the current mode, draws.
    pos = Positions(sc, win)
    draw_axes(win, pos, mode)
    return


# Perform one draw window cycle.
def cycle(sc, win, keypress, interval, modes):
    # Refresh states.
    interval.update()
    keypress.read(win)
    if keypress.key == curses.KEY_RESIZE:
        win.erase()
    if keypress.mode_changed:
        win.erase()
        keypress.mode_changed = False
    
    # Get the index of the active mode, using the index of it's key.
    index_of_active_mode = mode_keys.index(keypress.current_mode)
    # 'modes' is a list of Mode objects that configure a window.
    active_mode = modes[index_of_active_mode]


    # Get block data
    if interval.ready_to_call_block:
        pass # get_block()
    
    # Draw graph
    draw_graph(sc, win, active_mode)

    return keypress.active


def begin(sc):
    # Creates a curses window in terminal and main tracking objects,
    # starts rendering in a loop that reacts to keypresses.

    h, w = sc.getmaxyx()  
    win = curses.newwin(h, w, 0, 0) 
    win.keypad(1)  
    curses.curs_set(0) 
    starting_key = mode_keys[0]
    keypress = Keypress(starting_key)
    interval = Interval()
    modes = [Mode(key) for key in mode_keys]

    # Begin main program loop.
    active = True
    while active:
        win.border(0)
        win.timeout(100)
        active = cycle(sc, win, keypress, interval, modes)
       
    # Close program.
    h, w = sc.getmaxyx()
    sc.addstr(h//2, w//2, 'Bye-bye!')
    sc.refresh()
    time.sleep(2)
    curses.endwin()

curses.wrapper(begin)