import curses
import time
from enum import Enum
import requests
import time
from itertools import count

# Change this to the address of your node.
node = "http://127.0.0.1:8545"

ModeNames = Enum('Modes', 'LATEST_FEES BASE_FEE')
mode_keys = [ord('1'), ord('2')]
price_string = 'nanoether per gas (Gwei per gas)'
mode_params = {
    ModeNames.LATEST_FEES: {
        'mode_key': mode_keys[0],
        'graph_title': 'Fees for latest block',
        'x_axis_name': 'Transaction index',
        'y_axis_name': price_string,
        'block_depth': 1,
        'block_data': ['gasLimit', 'gasUsed', 'number', 'timestamp',
            'baseFeePerGas'],
        'get_transactions': True,
        'transaction_params': ['type','gasPrice','transactionIndex',
            'maxFeePerGas','maxPriorityFeePerGas'],
        'sets_to_graph': [
            {'name': 'maxPriorityFeePerGas', 
            'symbol': '^', 
            'type': 'transactions', 
            'x': 'transactionIndex',
            'y': 'maxPriorityFeePerGas'},
            {'name': 'maxFeePerGas', 
            'symbol': '*', 
            'type': 'transactions', 
            'x': 'transactionIndex',
            'y': 'maxFeePerGas'},
            {'name': 'gasPrice', 
            'symbol': '#', 
            'type': 'transactions', 
            'x': 'transactionIndex',
            'y': 'gasPrice'}
        ],
        'y_display_scale': 10**9
    },
    ModeNames.BASE_FEE: {
        'mode_key': mode_keys[1],
        'graph_title': 'Recent base fees',
        'x_axis_name': 'Block number',
        'y_axis_name': price_string,
        'block_depth': 100 
    }
}
# Maintains the sequence of JSON-RPC API calls.
block_call_ids = count()

def hex_to_int(hex_string):
    # Hex parser that handles 'None' type.
    if hex_string:
        return int(hex_string, 16)
    else:
        return None

def parse_block(block, mode):
    # Retrieves only the relevant data for a mode. 
    # If a desired field is absent, key will have 'None' value.
    single_block = {
        key: hex_to_int(block.get(key))
        for key in mode.params['block_data'] 
    }
    transactions = [
        {
            key: hex_to_int(block_tx.get(key)) 
            for key in mode.params['transaction_params']
        }       
        for block_tx in block['transactions']
        if mode.params['get_transactions']
    ]
    single_block['transactions'] = transactions
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
    # Returns the ModeName for a given keypress. 
    # ord('1') corresponding to the first (default) mode.
    for mode_name in ModeNames:
        if keypress == mode_params[mode_name]['mode_key']:
            return mode_name

    
class BlockDataManager:
    # Holds data. Modes query the manager for data.
    # Periodically refreshes by calling node.
    def __init__(self, modes):
        # list of block objects.
        self.recent_blocks_parsed = []
        # When initialised, only get one block (for speed)
        self.latest_block_transactions = get_blocks(
            ['latest'], modes[0])

    def get_all_data(self, modes):
        # Retrieves the data required for each mode.
        # Modes specify the raw data they will need.
        # API is called so that any mode will have the data ready.
        self.latest_block_transactions = get_blocks(
            ['latest'], modes[0])
        # TODO Call node to get data and save as cache.
        # self.recent_base_fees = xyz
        # self.recent_priority/maxfees = xyz
        # self.latest_block_transactions = xyz
        pass

def format_set(data, format):
    # Accepts block data and produces values in standard format.
    # set_name, set_symbol, x_list, y_list.
    if format['type'] == 'transactions':
        x = [t[format['x']]for t in data[0]['transactions']]
        y = [t[format['y']]for t in data[0]['transactions']]
        return {
            'set_name':format['name'],'set_symbol':format['symbol'],
            'x_list': x, 'y_list': y} 
        
class Mode:
    # The graph context (data, window display)
    # Local 'mode' is a Mode object that configures a display.
    def __init__(self, keypress):
        self.name = get_mode_from_key(keypress)
        # Get unique mode features from global paramater config dict.
        self.params = mode_params[self.name]
        self.data = None

    def prepare_data(self, data):
        # Accepts data, uses self.params to select and refine.
        # Saves a list of sets of points to be graphed, with 
        self.data = [
            format_set(data, format)
            for format in self.params['sets_to_graph']]

    def define_chunks(self):
        if self.name == ModeNames.BASE_FEE:
            start = 5072606
            # start = 5062605  # Goerli fork block. 
            self.start_block = start # a goerli block.
            self.end_block = start + 1000
            self.block_chunks = 100
        return self


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
        self.time = int(time.time())
    
    def update(self):
        self.sec_since_call =  int(time.time()) - self.time
        if self.sec_since_call >= 6:
            self.ready_to_call_block = True
            self.reset()
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
    [
        win.addstr(pos.border + i, pos.w - pos.border - 3 - \
                len(set['set_name']),
            f"{set['set_symbol']} - {set['set_name']}")
        for i, set in enumerate(mode.data)
    ]
    # x-axis
    win.hline(pos.x_axis_base[0], pos.x_axis_base[1] + 1, 
        '_', pos.x_axis_width)
    x_name = mode.params['x_axis_name']
    win.addstr(pos.h - 2, 1 + \
        pos.x_axis_base[1] + pos.x_axis_width - len(x_name), x_name)
    # y-axis
    win.vline(pos.y_axis_tip[0], pos.y_axis_tip[1], 
        '|', pos.y_axis_height)
    win.addstr(1, pos.y_axis_tip[1] - 1, mode.params['y_axis_name'])
    # title
    title = mode.params['graph_title']
    win.addstr(pos.y_axis_tip[0] - 2,
        pos.w - pos.border - len(title) + 1, title)
    '''TODO. Add current block for context.
    block_str = f"Current block: {mode.?data['block_number']}"
    win.addstr(pos.y_axis_tip[0] - 1,
        #pos.w - pos.border - len(block_str) + 1, block_str)
    '''

def select_points(pos, mode):
    # Returns a tuple representing the first point to exclude,
    # and the number of subsequent points. Axis vals will be shifted
    # after these.
    number = pos.w - 2 * pos.border
    available_points = len(mode.data[0]['x_list'])
    if available_points > number:
        remove_num = available_points - number
        first_index = available_points // 2 - remove_num // 2
        return (first_index, remove_num)
    return (0, 0)


def get_min_xy_max_xy(mode):
    # Finds largest x and y vals of all plottable points.
    if mode.data[0]['x_list'] == []:
        return (0,0,0,0)
    min_x = min([
        min([0 if x==None else x for x in data_set['x_list']])
        for data_set in mode.data])
    min_y = min([
        min([0 if x==None else x for x in data_set['y_list']])
        for data_set in mode.data])
    max_x = max([
        max([0 if x==None else x for x in data_set['x_list']])
        for data_set in mode.data])
    max_y = max([
        max([0 if x==None else x for x in data_set['y_list']])
        for data_set in mode.data])
    return (min_x, min_y, max_x, max_y)

def scale_value(val, val_small_large, coord_small_large):
    # Scales the point onto the pixels available.
    # (val - src_min) / (src_range) * (dest_range) + dest_min.
    # Note: Curses maps yvals top down.
    src = val_small_large  # (min, max)
    dst = coord_small_large  # (minpix, maxpix)
    if src[1] == 0:
        return (0, 0)  # If none of the y vals > 0.
    s = (1 - (val - src[0]) / (src[1]-src[0])) * \
        (dst[1]-dst[0]) + dst[0]
    return int(s)


def get_coords(pos, x, y, min_xy_max_xy):
    # Calculates coordinates of an (x, y) point.
    # Applies a shift left if the value is right of the skipped vals.
    min_xpix = pos.border + 1
    # max_xpix = pos.w - pos.border - 1
    min_ypix = pos.border + 1
    max_ypix = pos.h - pos.border - 1
    _, miny, _, maxy = min_xy_max_xy
    x = x + min_xpix
    y = scale_value(y, (miny, maxy), (min_ypix, max_ypix))
    assert x >= 0 and y >= 0
    return (x, y)


def plot_set(win, pos, set, skip, min_xy_max_xy):
    # Plots points of a set on an axis. 
    # Removes skipped middle points.
    skip_end = skip[0] + skip[1]
    for i in range(len(set['x_list'])):
        shift = 0
        if i in range(skip[0], skip_end):
            continue
        if i >= skip_end:
            shift = skip[1]
        xval, yval = (set['x_list'][i], set['y_list'][i])
        if yval is not None:
            x, y = get_coords(pos, i - shift, yval, min_xy_max_xy) 
            # Plot point
            win.addstr(y , x, set['set_symbol'])


def draw_scales(win, pos, mode, min_xy_max_xy, points_to_skip):
    # Places values on the axes.
    # Add y values to axes, high to low.    
    n = 4  # number of notches to display y values at.
    for i in range(n):
        y_height = pos.y_axis_tip[0] + i * pos.y_axis_height // n
        y_val = str((n-i) * min_xy_max_xy[3] // n // \
            mode.params['y_display_scale'])
        win.addstr(y_height, pos.y_axis_tip[1] - 3, y_val)
    # Move the max axis if the axis is not fully filled.
    filler = max(0, pos.x_axis_width - \
        len(mode.data[0]['x_list']))
    # Add x axis max. 
    x_dist = pos.w - pos.border - len(str(min_xy_max_xy[2])) - filler
    win.addstr(pos.h - pos.border + 1, x_dist + 1, 
        str(min_xy_max_xy[2]))
    # Add x axis half, accounting for skipped middle values.
    x_dist = (x_dist + len(str(min_xy_max_xy[2])) - \
        pos.border) // 2 
    skip = 0
    if points_to_skip[1] > 0:
        skip = points_to_skip[0]
        x_range = min_xy_max_xy[2] - min_xy_max_xy[0]
        half_val = min_xy_max_xy[2] * skip // x_range    
        hidden_str = f'|| Middle {points_to_skip[1]} points hidden.'
        win.addstr(2, x_dist, hidden_str)
    else:
        half_val = min_xy_max_xy[2] // 2
    win.addstr(pos.h - pos.border + 1, x_dist - 1, str(half_val))


def draw_points(win, pos, mode):
    # Takes data, creates scale, fits to window, graphs.
    points_to_skip = select_points(pos, mode)
    min_xy_max_xy = get_min_xy_max_xy(mode)
    draw_scales(win, pos, mode, min_xy_max_xy, points_to_skip)
    [
        plot_set(win, pos, set, points_to_skip, min_xy_max_xy) 
        for set in mode.data
    ]


def draw_graph(sc, win, mode):
    # Gets positions of elements for the current mode, draws.
    pos = Positions(sc, win)

    draw_points(win, pos, mode)

    draw_axes(win, pos, mode)
    return


def detect_keypress(win, keypress):
    # Reacts to either window being resized or keyboard activity.
    keypress.read(win)
    if keypress.key == curses.KEY_RESIZE:
        win.erase()
    if keypress.mode_changed:
        win.erase()
        keypress.mode_changed = False


def cycle(sc, win, keypress, interval, modes, data_manager): 
    # Perform one draw window cycle.
    interval.update()
    detect_keypress(win, keypress)
    # Get mode define by keyboard number input.
    mode = modes[mode_keys.index(keypress.current_mode)]
    if interval.ready_to_call_block:
        data_manager.get_all_data(modes)
        modes[0].prepare_data(data_manager.latest_block_transactions)
        win.erase()
    #if interval.ready_to_call_block:
    draw_graph(sc, win, mode)
    return keypress.active


def main(sc):
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
    data_manager = BlockDataManager(modes)
    
    # Start display for first mode for snappy beginning.
    modes[0].prepare_data(data_manager.latest_block_transactions)
    draw_graph(sc, win, modes[0]) 
    # Start grabbing data for inactive modes.
    data_manager.get_all_data(modes)

    # Begin main display loop.
    active = True
    while active:
        win.border(0)
        win.timeout(100)
        active = cycle(sc, win, keypress, interval, modes, 
            data_manager)
       
    # Close program.
    h, w = sc.getmaxyx()
    goodbye = 'Bye Bye! Thanks for playing Fee-Feed'
    sc.addstr(h//2, w//2 - len(goodbye)//2, goodbye)
    sc.refresh()
    time.sleep(2)
    curses.endwin()


if __name__=="__main__":
    curses.wrapper(main)