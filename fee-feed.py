import curses
from re import U
import time
from enum import Enum
import requests
import time
from itertools import count

ModeNames = Enum('Modes', 'BASE_FEE PRIORITY_FEE')
mode_keys = [ord('1'), ord('2')]

# Which blocks to include

node = "http://127.0.0.1:8545"
block_call_ids = count()


def parse_block(block, mode):
    # Filters a block to retrieve only the relevant data
    # for that mode.
    single_block = {}
    for param in mode.param_list:
        single_block[param] = int(block[param], 16)
    return single_block


def get_blocks(blocks, mode):
    # Calls a node and asks for blocks in chunks.
    batch = [
        {
            "jsonrpc": "2.0",
            "id": next(block_call_ids),
            "method": "eth_getBlockByNumber",
            "params": [f"{block}", True],
        }
        for block in blocks
    ]
    responses = requests.post(node, json=batch).json()

    return [parse_block(res["result"], mode) for res in responses]

def get_data_batches(mode):
    # Organises desired data into groups for batch calling.
    blocks = []
    for chunk in range(mode.start_block, mode.end_block, 
        mode.block_chunks):
        block_list = get_blocks(
            range(chunk, 
                min(mode.end_block + 1, chunk + mode.block_chunks)),
            mode)
        [blocks.append(block) for block in block_list]
    return blocks


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
        self.graph_points_x = []
        self.graph_points_y = []
        self.coords_x = []
        self.coords_y = []
        self.start_block = None
        self.end_block = None
        self.block_chunks = None

    def define_chunks(self):
        if self.current == ModeNames.BASE_FEE:
            start = 5072606
            # start = 5062605  # Goerli fork block. 
            self.start_block = start # a goerli block.
            self.end_block = start + 1000
            self.block_chunks = 100
        return self

    def define_parse_params(self):
        if self.current == ModeNames.BASE_FEE:
             self.param_list = ['number','baseFeePerGas']
        

    def get_data(self):
        self.define_chunks()
        self.define_parse_params()
        self.data = get_data_batches(self)


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
    win.addstr(pos.h - 1, 
        pos.x_axis_base[1] + pos.x_axis_width - len(x_name), mode.x_name)
    # y-axis
    win.vline(pos.y_axis_tip[0], pos.y_axis_tip[1], 
        '|', pos.y_axis_height)
    win.addstr(1, pos.y_axis_tip[1] - 2, mode.y_name)
    # title
    win.addstr(pos.y_axis_tip[0] - 2, 
        pos.w - pos.border - len(mode.graph_name), mode.graph_name)
    return 

def select_points(pos, mode):
    # From the data chooses points that will be able to fit.
    number = pos.w - 2 * pos.border
    return mode.data[:number]

def plot_point(win, mode, index, symbol):
    # Plots a single point.
    y = int(mode.coords_y[index])
    x = mode.coords_x[index]
    win.addstr(y, x, symbol)

def get_scale(points, mode):
    # Attaches points to tracking object, gets max vals.
    for point in points:
        xval = point[mode.param_list[0]]
        mode.graph_points_x.append(xval)
        yval = point[mode.param_list[1]]
        mode.graph_points_y.append(yval)
    ymax = max(mode.graph_points_y)
    xmax = max(mode.graph_points_x)
    return (ymax, xmax)

def scale_value(val, src, dst):
    # Scales the point onto the pixels available.
    # Inverting the scale, because Curses maps yvals top down.

    return (1 - (val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def get_coordinates(pos, mode, scale):
    # Calculates coordinates for each point.
    (ymax, xmax) = scale
    y_pix_max = pos.h - pos.border - 1
    y_pix_min = pos.border + 1 
    for index in range(len(mode.graph_points_x)):
        # Space out x values evenly left to right
        coord = pos.border + 2 + index
        mode.coords_x.append(coord)

    for point in mode.graph_points_y:
        # Map the y values to the axis.
        from_range = (0, ymax)
        to_range =  (y_pix_min, y_pix_max)
        coord = scale_value(point, from_range, to_range)
        mode.coords_y.append(coord)

def draw_points(win, pos, mode):
    # Takes data, creates scale, fits to window, graphs.
    points = select_points(pos, mode)
    scale = get_scale(points, mode)
    # Add max values to axes.
    win.addstr(pos.y_axis_tip[0], pos.y_axis_tip[1] - 3, str(scale[0]))
    win.addstr(pos.h - pos.border + 1, 
        pos.w - pos.border - len(str(scale[1])), str(scale[1]))

    get_coordinates(pos, mode, scale)

    [
        plot_point(win, mode, index, '*') 
        for index in range(len(points))
    ]


def draw_graph(sc, win, mode):
    # Gets positions of elements for the current mode, draws.
    pos = Positions(sc, win)

    draw_points(win, pos, mode)

    draw_axes(win, pos, mode)
    return


def cycle(sc, win, keypress, interval, modes): 
    # Perform one draw window cycle.

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
    mode = modes[index_of_active_mode]


    # Get block data
    if interval.ready_to_call_block:
        mode.get_data()
        # Draw graph
        draw_graph(sc, win, mode)

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