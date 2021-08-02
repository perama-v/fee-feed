import curses
import time
from enum import Enum
import requests
import time
from itertools import count, accumulate
from bisect import bisect_left
import statistics
import os
from dotenv import load_dotenv

load_dotenv()

# Change this to the address of your node.
node = os.environ.get('RPC_URL')

ModeNames = Enum('Modes', 'LATEST_FEES BOXPLOT BOXPLOT_NO_OUTLIERS \
FULLNESS SPREAD BURN')
mode_keys = [ord('1'), ord('2'), ord('3'), ord('4'),
    ord('5'), ord('6')]
price_string = 'nanoether per gas (Gwei per gas)'
oldest_block_depth = 200
mode_params = {
    ModeNames.LATEST_FEES: {
        'mode_key': mode_keys[0],
        'button' : '1',
        'graph_title': 'Fees for latest block',
        'x_axis_name': 'Transaction index',
        'y_axis_name': price_string,
        'oldest_required': 1,
        'block_data': ['gasLimit', 'gasUsed', 'number', 'timestamp',
            'baseFeePerGas'],
        'get_transactions': True,
        'transaction_params': ['type','gasPrice','transactionIndex',
            'maxFeePerGas','maxPriorityFeePerGas'],
        'sets_to_graph': [
            {'name': 'maxPriorityFeePerGas',
            'symbol': '°',
            'loc_for_set': 'transactions',
            'x': 'transactionIndex',
            'y': 'maxPriorityFeePerGas'},
            {'name': 'maxFeePerGas',
            'symbol': '═',
            'loc_for_set': 'transactions',
            'x': 'transactionIndex',
            'y': 'maxFeePerGas'},
            {'name': 'gasPrice',
            'symbol': '╦',
            'loc_for_set': 'transactions',
            'x': 'transactionIndex',
            'y': 'gasPrice'}
        ],
        'set_plot_order': [0, 1, 2],
        'y_display_scale': 10**9,
        'loc_in_manager': 'latest_block_transactions'
    },
    ModeNames.BOXPLOT: {
        'mode_key': mode_keys[1],
        'button': '2',
        'graph_title': 'Recent priority fees',
        'x_axis_name': 'Block number',
        'y_axis_name': price_string,
        'oldest_required': oldest_block_depth,
        'block_stats': ['base_fee','Q4','Q3','Q2','Q1','Q0'
            'mode_priority'],
        'transaction_params': ['gasPrice','maxFeePerGas',
            'maxPriorityFeePerGas'],
        'sets_to_graph': [
            {'name': 'max',
            'symbol': '*',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'max'},
            {'name': 'Q4 ',
            'symbol': '┬',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q4'},
            {'name': 'Q3 ',
            'symbol': '▓',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q3'},
            {'name': 'med',
            'symbol': '▓',
            'x': 'block_number',
            'loc_for_set': 'statistics',
            'y': 'Q2'},
            {'name': 'Q1 ',
            'symbol': '▓',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q1'},
            {'name': 'Q0 ',
            'symbol': '┴',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q0'},
            {'name': 'min',
            'symbol': '*',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'min'},
            {'name': 'Base fee',
            'symbol': '≡',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'base_fee'},
        ],
        'set_plot_order': [0, 6, 1, 5, 3, 4, 2, 7],
        'y_display_scale': 10**9,
        'loc_in_manager': 'recent_blocks'
    },
    ModeNames.BOXPLOT_NO_OUTLIERS: {
        'mode_key': mode_keys[2],
        'button': '3',
        'graph_title': 'Recent priority fees - No outliers',
        'x_axis_name': 'Block number',
        'y_axis_name': price_string,
        'oldest_required': oldest_block_depth,
        'block_stats': ['Q4','Q3','Q2','Q1','Q0'],
        'transaction_params': ['maxFeePerGas',
            'maxPriorityFeePerGas'],
        'sets_to_graph': [
            {'name': 'Q4 ',
            'symbol': '┬',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q4'},
            {'name': 'Q3 ',
            'symbol': '▓',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q3'},
            {'name': 'med',
            'symbol': '▓',
            'x': 'block_number',
            'loc_for_set': 'statistics',
            'y': 'Q2'},
            {'name': 'Q1 ',
            'symbol': '▓',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q1'},
            {'name': 'Q0 ',
            'symbol': '┴',
            'loc_for_set': 'statistics',
            'x': 'block_number',
            'y': 'Q0'}
        ],
        'set_plot_order': [0, 4, 2, 3, 1],
        'y_display_scale': 10**9,
        'loc_in_manager': 'recent_blocks'
    },
    ModeNames.FULLNESS: {
            'mode_key': mode_keys[3],
            'button': '4',
            'graph_title': 'Block utilisation and other measures',
            'x_axis_name': 'Block number',
            'y_axis_name': 'Scalar value (see legend)',
            'oldest_required': oldest_block_depth,
            'block_stats': ['gas_utilisation', 'type_2_utilisation',
            'lowest_nonzero_fee'],
            'sets_to_graph': [
                {'name': 'Gas utilisation',
                'symbol': '░',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'gas_utilisation'},
                {'name': 'Type 2 utilisation',
                'symbol': '²',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'type_2_utilisation'},
                {'name': 'lowest nonzero fee (nanoether)',
                'symbol': '╩',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'lowest_nonzero_fee'},
                ],
            'set_plot_order': [0, 1, 2],
            'y_display_scale': 1,
            'loc_in_manager': 'recent_blocks'
        },
    ModeNames.SPREAD: {
            'mode_key': mode_keys[4],
            'button': '5',
            'graph_title': 'Fee spread and other measures',
            'x_axis_name': 'Block number',
            'y_axis_name': 'Scalar value (see legend)',
            'oldest_required': oldest_block_depth,
            'block_stats': ['priority_std',
                'average_max_multiple', 'base_fee'],
            'sets_to_graph': [
                {'name': 'Priority standard deviation',
                'symbol': 'σ',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'priority_std'},
                {'name': 'Average max fee / basefee',
                'symbol': '×',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'average_max_multiple'},
                {'name': 'Base fee',
                'symbol': '≡',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'base_fee'},
            ],
            'set_plot_order': [0, 1, 2],
            'y_display_scale': 10**9,
            'loc_in_manager': 'recent_blocks'
        },
    ModeNames.BURN: {
            'mode_key': mode_keys[5],
            'button': '6',
            'graph_title': 'Ether burned for recent blocks',
            'x_axis_name': 'Block number',
            'y_axis_name': 'picoether burned',
            'oldest_required': oldest_block_depth,
            'block_stats': ['block_burn', 'cumulative_burn'],
            'sets_to_graph': [
                {'name': 'Block burn',
                'symbol': '¤',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'block_burn'},
                {'name': 'Recent cumulative burn',
                'symbol': 'ð',
                'loc_for_set': 'statistics',
                'x': 'block_number',
                'y': 'cumulative_burn'},
            ],
            'set_plot_order': [1, 0],
            'y_display_scale': 10**6,
            'loc_in_manager': 'recent_blocks'
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


def get_transactions_from_block(block, mode):
    # Uses parameter list to get important fields from block.
    transactions = [
        {
            key: hex_to_int(block_tx.get(key))
            for key in mode.params['transaction_params']
        }
        for block_tx in block['transactions']
        if mode.params['get_transactions']
    ]
    return transactions


def parse_block(block, mode):
    # Retrieves only the relevant data for a mode.
    # If a desired field is absent, key will have 'None' value.
    single_block = {
        key: hex_to_int(block.get(key))
        for key in mode.params['block_data']
    }
    transactions = get_transactions_from_block(block, mode)
    single_block['transactions'] = transactions
    return single_block


def infer_priority_fee(transaction, base_fee):
    # Get priority fee equivalent for all tx types, even legacy.
    # Inferred priority == effective miner fee.
    inferred_p_f = 0
    if 'maxPriorityFeePerGas' not in transaction.keys():
        inferred_p_f = int(transaction['gasPrice'], 16) - base_fee
    else:
        inferred_p_f = int(transaction['maxPriorityFeePerGas'], 16)
    transaction['inferred_priority'] = inferred_p_f
    return transaction


def calculate_priority_boxplot(block, with_inferred_priority, result):
    # Generates values used for box plot, adds to result.
    # Sort tx list by inferred priority fee.
    # TODO consider excluding zero priority fee transactions.
    tx_by_fee = sorted(with_inferred_priority,
        key = lambda tx: tx['inferred_priority'])

    # Make a list of accumulating gas.
    tx_gas_used = [
        int(tx['gasUsed'], 16)
        for tx in with_inferred_priority
    ]
    gas_clock = list(accumulate(tx_gas_used))
    block_gas = int(block['gasUsed'], 16)
    result['block_gas'] = block_gas
    percentiles = {
        'min': 0,
        'Q1': 25,
        'Q2': 50,
        'Q3': 75,
        'max': 100
    }
    for name, percentile in percentiles.items():
        # Use gas_clock for the index of the transaction
        # once the gas percentile is passed.
        tx_index = bisect_left(gas_clock, percentile*block_gas/100)
        fee = tx_by_fee[tx_index]['inferred_priority']
        result[name] = fee

    # Lower and upper outlier thresholds.
    interquartile_range = result['Q3'] - result['Q1']
    outlier_upper = int(result['Q2'] + 1.5 * interquartile_range)
    outlier_lower = int(result['Q2'] - 1.5 * interquartile_range)

    # Define upper and lower whiskers based on presence of outliers.
    if result['min'] < outlier_lower:
        result['Q0'] = outlier_lower
    else:
        result['Q0'] = result['min']

    if result['max'] > outlier_upper:
        result['Q4'] = outlier_upper
    else:
        result['Q4'] = result['max']
    return result


def calculate_gas_utilisation(block, result):
    # Percentage of possible gas used by a block
    # range: [0-100], base fee targets 50.
    used = int(block['gasUsed'], 16)
    limit =  int(block['gasLimit'], 16)
    result['gas_utilisation'] = int(100 * used / limit)
    return result


def calculate_lowest_nonzero_fee(sorted_fees, result):
    # Finds the lowest fee the miner accepted that was > 0.
    # To excluded miner-originated / MEV transactions.
    # Returns as nanoether (10**9 wei), unlike other fields.
    min = sorted_fees[0]
    for fee in sorted_fees:
        if fee > 0:
            min = fee
            break
    result['lowest_nonzero_fee'] = int(min / 10**9)
    return result


def calculate_priority_std(inferred_fees, result):
    # std of the max priority fee for a block.
    std = 0
    if len(inferred_fees) > 1:
        std = int(statistics.stdev(inferred_fees))
    result['priority_std'] = std
    return result


def calculate_type_2_utilisation(block, result):
    # Percentage of transactions that are type 0x2 (London).
    # range: [0-100]
    num = 0
    for t in block['transactions']:
        if t['type'] == '0x2':
            num += 1
    percentage = 100 * num // len(block['transactions'])
    result['type_2_utilisation'] = percentage
    return result


def calculate_average_max_multiple(block, fees, result):
    # The relative value of the average max fee vs the base fee.
    # range: [1-any]. 5 = max fees are 5x base fee.
    # 1.2 = max fees are 20% higher than the base fee.
    multiple = 0
    if 'baseFeePerGas' in block.keys():
        mean = statistics.mean(fees)
        multiple = int(mean / int(block['baseFeePerGas'], 16))
    result['average_max_multiple'] = multiple
    return result


def calculate_block_burn(block, result):
    # Accumulated burn over the currently displayed block range.
    burn = 0
    if 'baseFeePerGas' in block.keys():
        burn = int(block['gasUsed'], 16) * \
            int(block['baseFeePerGas'], 16)
    result['block_burn'] = burn
    return result


def block_analysis(block):
    # Gets information about fee, e.g. base fee and some
    # gas-based percentiles of effective miner fee.
    result = {}
    if len(block['transactions']) == 0:
        return None

    # Get base fee.
    base_fee = 0
    if 'baseFeePerGas' in block.keys():
        base_fee = int(block['baseFeePerGas'], 16)
    result['base_fee'] = base_fee


    with_inferred = [
        infer_priority_fee(transaction, base_fee)
        for transaction in block['transactions']]
    fees = sorted([
        block['inferred_priority'] for block in with_inferred])

    result = calculate_priority_boxplot(block, with_inferred, result)
    result = calculate_lowest_nonzero_fee(fees, result)
    result = calculate_gas_utilisation(block, result)
    result = calculate_priority_std(fees, result)
    result = calculate_type_2_utilisation(block, result)
    result = calculate_average_max_multiple(block, fees, result)
    result = calculate_block_burn(block, result)

    # Add new data functions here.
    # E.g., result = get_special_metric(block, result)

    result['block_number'] = int(block['number'], 16)
    return result


def get_receipts(block):
    # Accepts a block from getBlockByNumber
    # Updates the transactions to include receipt data.
    transactions = block['transactions']
    if len(transactions) == 0:
        return block
    batch = [
        {
            "jsonrpc": "2.0",
            "id": next(block_call_ids),
            "method": "eth_getTransactionReceipt",
            "params": [f"{t['hash']}"],
        }
        for t in transactions
    ]
    responses = requests.post(node, json=batch).json()
    for index, res in enumerate(responses):
        for key, val in res['result'].items():
            transactions[index][key] = val
    block['transactions'] = transactions
    return block
     

def get_blocks(blocks, mode=None):
    # Calls a node and asks for blocks in chunks.

    batch = [
        {
            "jsonrpc": "2.0",
            "id": next(block_call_ids),
            "method": "eth_getBlockByNumber",
            "params": [f"{str(hex(block))}", True],
        }
        for block in blocks
    ]
    responses = requests.post(node, json=batch).json()
    blocks_with_receipts = [
        get_receipts(res["result"])
        for res in responses
    ]
    if mode is None:
        return [
            block_analysis(block)
            for block in blocks_with_receipts
            ]
    else:
        return [
            parse_block(block, mode)
            for block in blocks_with_receipts
            ]


def get_block_number():
    # Returns the latest block number (integer)
    get_num = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_blockNumber",
        "params": [],
        }
    res = requests.post(node, json=get_num).json()
    current = int(res["result"], 16)
    return current


def new_block_exists(known_block):
    # This function asks node for the current block
    # Returns True if higher than known block.
    current = get_block_number()
    if current > known_block:
        return True
    return False


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


def get_latest_only(modes):
    # Returns the latest block for quick displays
    latest = get_block_number()
    block = get_blocks([latest], modes[0])[0]
    return block, block['number']


def most_distant_needed(modes):
    # Finds the number of recent blocks needed at all times.
    return max([mode.params['oldest_required'] for mode in modes])

def get_cumulative_burn(block_stats):
    # Accepts up to date list of analysed block data.
    # Sums wei burned, adds total to each block incrementally.
    current_total = 0
    ordered = sorted(block_stats, key = lambda x: x['block_number'])
    with_burn_total = []
    for block in ordered:
        current_total += block['block_burn']
        block['cumulative_burn'] = current_total
        with_burn_total.append(block)
    return with_burn_total

class BlockDataManager:
    # Holds data. Modes query the manager for data.
    # Periodically refreshes by calling node.
    def __init__(self, modes):
        self.all_data = {
            m.params['loc_in_manager']: None for m in modes
        }
        # When initialised, only get one block (for speed)
        (block, num) = get_latest_only(modes)
        # Store for data for first mode to use.
        first_mode = modes[0]
        loc = first_mode.params['loc_in_manager']
        self.all_data[loc] = block
        self.oldest_block = num  # Oldest for which tx data is held.
        self.current_block = num  # Newest for which tx data is held.
        self.range_required = most_distant_needed(modes)
        self.modes_available = [first_mode.name]

    def get_missing_blocks(self, newest_aware_of, modes):
        # Maintenance data management. Refreshes recent block list.
        # Assesses if block data needs are met, retrieves if needed.
        # Will only get a subset if there is a large range missing.
        max_update = oldest_block_depth  # Can limit update size.

        oldest_needed = newest_aware_of - self.range_required
        needed = [i for i in range(oldest_needed,
            newest_aware_of)]
        # Remove outdated blocks.
        cached = self.all_data['recent_blocks']
        relevant = []
        if cached is not None:
            cached = cached['statistics']
            relevant = [
                block for block in cached
                if block['block_number'] in needed
            ]
        # Fetch missing blocks.
        have_block_nums = [
            block['block_number'] for block in relevant
        ]
        missing_numbers = [
            block_num for block_num in needed
            if not block_num in have_block_nums
        ]
        if len(missing_numbers) > max_update:
            missing_numbers = missing_numbers[-max_update:]
        if len(missing_numbers) > 0:
            retrieved = get_blocks(missing_numbers, mode=None)
            [
                relevant.append(block)
                for block in retrieved
                if block is not None
            ]
        # Add burn accumulated over current range.
        relevant_with_burn = get_cumulative_burn(relevant)
        new_stats = {'statistics': relevant_with_burn}


        self.all_data['recent_blocks'] = new_stats

        # TODO: Handle reorgs.
        # E.g., Starting from recent, walk the hashes and discard
        # if not in chain, then refill any missing.
        self.current_block = newest_aware_of
        # Make all modes available.
        self.modes_available = [m.name for m in modes]

    def get_first_mode_data(self, modes):
        # Gets data from latest block for fast display.
        block, block_num = get_latest_only(modes)
        self.all_data['latest_block_transactions'] = block
        self.current_block = block_num


def format_set(data, format):
    # Accepts block data and produces values in standard format.
    # set_name, set_symbol, x_list, y_list.
    set_data = data[format['loc_for_set']]
    a=1
    x = [t[format['x']]for t in set_data]
    y = [t[format['y']]for t in set_data]
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
        # A list of data dicts with x, y, name, symbol.
        self.data = None
        self.current_block = None

    def prepare_data(self, data_manager):
        # Accepts manager, uses self.params to select and refine.
        # Saves a list of sets of points to be graphed.
        data = data_manager.all_data[self.params['loc_in_manager']]
        self.data = [
            format_set(data, format)
            for format in self.params['sets_to_graph']]
        self.current_block = data_manager.current_block


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

    def read(self, win, data_manager):
        # Reads last key pressed, detects changes.
        self.key = win.getch()
        if self.key == ord('q'):
            self.active = False

        available_mode_keys = [
            mode_params[m]['mode_key']
            for m in data_manager.modes_available
        ]
        # If valid key
        if self.key in mode_keys:
            if self.key in available_mode_keys:
                self.modify_mode()


class Interval:
    # Determines if it is the right time to get data.
    def __init__(self):
        self.sec_since_call = 0
        delay_ms = 400  # Delay after first window display.
        self.time_msec_after_start = int(time.time()*1000) + delay_ms
        self.time = int(time.time())
        self.ready_to_call_block = False
        # A small delay to allow the interface to appear.
        self.ready_for_startup_data = False
        self.startup_data_done = False

    def reset(self):
        self.time = int(time.time())

    def startup_data_retrieved(self):
        self.ready_for_startup_data = False
        self.startup_data_done = True

    def update(self, current_block_num):
        self.sec_since_call =  int(time.time()) - self.time
        if not self.startup_data_done:
            if int(time.time()*1000) > self.time_msec_after_start:
                self.ready_for_startup_data = True
        if self.sec_since_call >= 3:
            if new_block_exists(current_block_num):
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
    block_str = f"Current block: {mode.current_block}"
    win.addstr(1, pos.w - pos.border - len(block_str) + 1, block_str)


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
        return 0  # If none of the y vals > 0.
    if src[0] == src[1]:
        return int(src[0]) # If there is a single y value.
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
    yscale = mode.params['y_display_scale']
    for i in range(n):
        y_height = pos.y_axis_tip[0] + i * pos.y_axis_height // n
        y_val = str((n-i) * min_xy_max_xy[3] // n // yscale)
        win.addstr(y_height, pos.y_axis_tip[1] - 3, y_val)
    # Draw y minimum
    win.addstr(pos.x_axis_base[0] - 1, pos.x_axis_base[1] - 3,
        str(int(min_xy_max_xy[1] / yscale)))

    # Move the max axis if the axis is not fully filled.
    filler = max(0, pos.x_axis_width - \
        len(mode.data[0]['x_list']))
    # Add x axis max.
    x_dist = pos.w - pos.border - len(str(min_xy_max_xy[2])) - filler
    win.addstr(pos.h - pos.border + 1, x_dist + 1,
        str(min_xy_max_xy[2]))
    # Add x axis mid-point, accounting for skipped middle values.
    x_dist = (x_dist + len(str(min_xy_max_xy[2])) - \
        pos.border) // 2
    skip = 0
    min_x = min_xy_max_xy[0]
    x_range = min_xy_max_xy[2] - min_x
    if points_to_skip[1] > 0:
        skip = points_to_skip[0]
        half_val = x_range * skip // x_range + min_x
        hidden_str1 = f'|...| points {skip} to '
        hidden_str2 = f'{skip + points_to_skip[1]} hidden.'
        win.addstr(3, x_dist, hidden_str1+hidden_str2)
    else:
        half_val = x_range // 2 + min_x
    win.addstr(pos.h - pos.border + 1, x_dist, str(half_val))
    # Add x axis minimum.
    win.addstr(pos.x_axis_base[0] + 1, pos.x_axis_base[1] + 1,
        str(min_x))


def draw_points(win, pos, mode):
    # Takes data, creates scale, fits to window, graphs.
    points_to_skip = select_points(pos, mode)
    min_xy_max_xy = get_min_xy_max_xy(mode)
    draw_scales(win, pos, mode, min_xy_max_xy, points_to_skip)
    [
        plot_set(win, pos, mode.data[i], points_to_skip, 
            min_xy_max_xy)
        for i in mode.params['set_plot_order']
    ]


def fetch_missing_and_prepare(data_manager, modes):
    # Investigates what data is missing, retrieves, then prepares.
    data_manager.get_missing_blocks(
        data_manager.current_block, modes)
    [m.prepare_data(data_manager) for m in modes]


def offer_modes(win, pos, mode, data_manager):
    # Shows the buttons that a user can press to select mode.
    current_button = mode_params[mode.name]['button']
    available = [
        mode_params[m]['button']
        for m in data_manager.modes_available
    ]
    highlighted = [
        f'[{m}]' if m == current_button else m
        for m in available
    ]
    mode_str = f'Modes: {" ".join(highlighted)}. '
    mode_str += 'Press num key or q to quit.'
    win.addstr(1, pos.w // 2 - len(mode_str)//2, mode_str)


def draw_graph(sc, win, mode, data_manager):
    # Gets positions of elements for the current mode, draws.
    pos = Positions(sc, win)
    if pos.w < pos.border * 12 or pos.h < pos.border * 2:
        msg = f'smol window'
        win.addstr(pos.h // 2, pos.w // 2 - len(msg) // 2, msg)
        return
    offer_modes(win, pos, mode, data_manager)
    if mode.data is None or len(mode.data[0]['x_list']) == 0:
        msg = f'Fetching block {mode.current_block}...'
        win.addstr(pos.h // 2, pos.w // 2 - len(msg) // 2, msg)
        return

    draw_points(win, pos, mode)
    draw_axes(win, pos, mode)
    return


def detect_window_and_keypress(win, keypress, data_manager):
    # Reacts to either window being resized or keyboard activity.
    keypress.read(win, data_manager)
    if keypress.key == curses.KEY_RESIZE:
        win.erase()
    if keypress.mode_changed:
        win.erase()
        keypress.mode_changed = False


def cycle(sc, win, keypress, interval, modes, data_manager):
    # Performs one draw window cycle.
    interval.update(data_manager.current_block)
    detect_window_and_keypress(win, keypress, data_manager)
    # Get mode define by keyboard number input.
    mode = modes[mode_keys.index(keypress.current_mode)]
    if interval.ready_to_call_block:
        # If a new block has been observed.
        data_manager.get_first_mode_data(modes)
        data_manager.get_missing_blocks(
            data_manager.current_block, modes)
        # Construct graphable representation.
        [m.prepare_data(data_manager) for m in modes]
        win.erase()
    if interval.ready_for_startup_data:
        fetch_missing_and_prepare(data_manager, modes)
        interval.startup_data_retrieved()
    draw_graph(sc, win, mode, data_manager)
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

    # Data for first mode for snappy beginning.
    modes[0].prepare_data(data_manager)

    # Begin main display loop.
    active = True
    while active:
        win.border(0)
        win.timeout(100)
        active = cycle(sc, win, keypress, interval, modes,
            data_manager)

    # Close program.
    h, w = sc.getmaxyx()
    goodbye = ['Bye!', '','ethworm.eth', 'If you want to support me', ]
    [
       sc.addstr(h//2 + 2 * i, w//2 - len(text)//2, text)
       for i, text in enumerate(goodbye)
    ]
    sc.refresh()
    time.sleep(3)
    curses.endwin()


if __name__=="__main__":
    curses.wrapper(main)
